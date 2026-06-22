use nalgebra::{DMatrix, DVector, SymmetricEigen};
use num_complex::Complex64;
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

type GradientRows = Vec<Vec<f64>>;
type PairGradients = Vec<GradientRows>;
type MemberGradients = Vec<PairGradients>;
type MemberResult = (Vec<f64>, PairGradients);

#[derive(Clone, Copy)]
enum FidelityMode {
    Real,
    Imag,
    Abs2,
}

impl FidelityMode {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "real" => Ok(Self::Real),
            "imag" => Ok(Self::Imag),
            "abs2" => Ok(Self::Abs2),
            _ => Err(PyValueError::new_err(
                "mode must be one of: abs2, imag, real",
            )),
        }
    }

    fn value(self, overlap: Complex64) -> f64 {
        match self {
            Self::Real => overlap.re,
            Self::Imag => overlap.im,
            Self::Abs2 => overlap.norm_sqr(),
        }
    }

    fn gradient(self, overlap: Complex64, derivative: Complex64) -> f64 {
        match self {
            Self::Real => derivative.re,
            Self::Imag => derivative.im,
            Self::Abs2 => 2.0 * (overlap.conj() * derivative).re,
        }
    }
}

fn matrix_from_slice(values: &[Complex64], rows: usize, cols: usize) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(rows, cols, values)
}

fn vector_from_slice(values: &[Complex64]) -> DVector<Complex64> {
    DVector::from_column_slice(values)
}

fn is_anti_hermitian(matrix: &DMatrix<Complex64>) -> bool {
    let scale = matrix
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if (matrix[(row, col)] + matrix[(col, row)].conj()).norm() > 1.0e-12 * scale {
                return false;
            }
        }
    }
    true
}

fn coherent_propagator_and_derivatives(
    generator: &DMatrix<Complex64>,
    controls: &[DMatrix<Complex64>],
    dt: f64,
) -> PyResult<(DMatrix<Complex64>, Vec<DMatrix<Complex64>>)> {
    if !is_anti_hermitian(generator) || controls.iter().any(|matrix| !is_anti_hermitian(matrix)) {
        return Err(PyValueError::new_err(
            "Rust gradient acceleration requires anti-Hermitian generators",
        ));
    }
    let i = Complex64::new(0.0, 1.0);
    let minus_i_dt = Complex64::new(0.0, -dt);
    let hermitian = generator * i;
    let decomposition = SymmetricEigen::new(hermitian);
    let vectors = decomposition.eigenvectors;
    let adjoint = vectors.adjoint();
    let eigenvalues = decomposition.eigenvalues;
    let dim = generator.nrows();
    let phases = DVector::from_iterator(
        dim,
        eigenvalues.iter().map(|value| (minus_i_dt * value).exp()),
    );
    let propagator = &vectors * DMatrix::from_diagonal(&phases) * &adjoint;

    let derivatives = controls
        .iter()
        .map(|control| {
            let rotated = &adjoint * (control * i) * &vectors;
            let mut weighted = DMatrix::zeros(dim, dim);
            let scale = eigenvalues
                .iter()
                .map(|value| value.abs())
                .fold(1.0_f64, f64::max);
            for row in 0..dim {
                for col in 0..dim {
                    let difference = eigenvalues[row] - eigenvalues[col];
                    let divided = if difference.abs() <= 1.0e-12 * scale {
                        minus_i_dt
                            * (minus_i_dt * (0.5 * (eigenvalues[row] + eigenvalues[col]))).exp()
                    } else {
                        (phases[row] - phases[col]) / difference
                    };
                    weighted[(row, col)] = divided * rotated[(row, col)];
                }
            }
            &vectors * weighted * &adjoint
        })
        .collect();
    Ok((propagator, derivatives))
}

#[allow(clippy::too_many_arguments)]
fn validate_shapes(
    drifts: &[Complex64],
    drift_shape: &[usize],
    operators: &[Complex64],
    operator_shape: &[usize],
    waveform: &[f64],
    waveform_shape: &[usize],
    rho_init: &[Complex64],
    init_shape: &[usize],
    rho_targ: &[Complex64],
    target_shape: &[usize],
) -> PyResult<(usize, usize, usize, usize)> {
    let (members, dim, dim_2) = (drift_shape[0], drift_shape[1], drift_shape[2]);
    let (op_members, channels, op_dim, op_dim_2) = (
        operator_shape[0],
        operator_shape[1],
        operator_shape[2],
        operator_shape[3],
    );
    let (steps, wfm_channels) = (waveform_shape[0], waveform_shape[1]);
    let (pairs, init_dim) = (init_shape[0], init_shape[1]);
    if members == 0 || channels == 0 || steps == 0 || pairs == 0 || dim == 0 {
        return Err(PyValueError::new_err(
            "all array dimensions must be non-zero",
        ));
    }
    if dim != dim_2 {
        return Err(PyValueError::new_err("drifts must contain square matrices"));
    }
    if op_members != members || op_dim != dim || op_dim_2 != dim {
        return Err(PyValueError::new_err(
            "operators must have shape (members, channels, dim, dim)",
        ));
    }
    if wfm_channels != channels {
        return Err(PyValueError::new_err(
            "waveform channel count must match operators",
        ));
    }
    if init_dim != dim || target_shape != init_shape {
        return Err(PyValueError::new_err(
            "rho_init and rho_targ must have shape (pairs, dim)",
        ));
    }
    if drifts.len() != members * dim * dim
        || operators.len() != members * channels * dim * dim
        || waveform.len() != steps * channels
        || rho_init.len() != pairs * dim
        || rho_targ.len() != pairs * dim
    {
        return Err(PyValueError::new_err("input arrays must be contiguous"));
    }
    Ok((members, channels, steps, dim))
}

/// Mean fidelity over a fully expanded ensemble of vector-state GRAPE problems.
///
/// Matrix assembly, slice exponentials, propagation, and ensemble reduction all
/// run in Rust. Ensemble members are evaluated in parallel with Rayon.
#[pyfunction]
fn grape_fidelity_vectors(
    drifts: PyReadonlyArray3<'_, Complex64>,
    operators: PyReadonlyArray4<'_, Complex64>,
    waveform: PyReadonlyArray2<'_, f64>,
    rho_init: PyReadonlyArray2<'_, Complex64>,
    rho_targ: PyReadonlyArray2<'_, Complex64>,
    dt: f64,
    mode: &str,
) -> PyResult<f64> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be finite and positive"));
    }
    let fidelity_mode = FidelityMode::parse(mode)?;
    let drift_shape = drifts.shape().to_vec();
    let operator_shape = operators.shape().to_vec();
    let waveform_shape = waveform.shape().to_vec();
    let init_shape = rho_init.shape().to_vec();
    let target_shape = rho_targ.shape().to_vec();
    let drifts = drifts.as_slice()?;
    let operators = operators.as_slice()?;
    let waveform = waveform.as_slice()?;
    let rho_init = rho_init.as_slice()?;
    let rho_targ = rho_targ.as_slice()?;
    let (members, channels, steps, dim) = validate_shapes(
        drifts,
        &drift_shape,
        operators,
        &operator_shape,
        waveform,
        &waveform_shape,
        rho_init,
        &init_shape,
        rho_targ,
        &target_shape,
    )?;
    let pairs = init_shape[0];
    let matrix_len = dim * dim;

    let sum: f64 = (0..members)
        .into_par_iter()
        .map(|member| {
            let drift_start = member * matrix_len;
            let drift = matrix_from_slice(&drifts[drift_start..drift_start + matrix_len], dim, dim);
            let member_operator_start = member * channels * matrix_len;
            let control_matrices: Vec<DMatrix<Complex64>> = (0..channels)
                .map(|channel| {
                    let start = member_operator_start + channel * matrix_len;
                    matrix_from_slice(&operators[start..start + matrix_len], dim, dim)
                })
                .collect();

            let propagators: Vec<DMatrix<Complex64>> = (0..steps)
                .map(|step| {
                    let mut generator = drift.clone();
                    for channel in 0..channels {
                        generator += &control_matrices[channel]
                            * Complex64::new(waveform[step * channels + channel], 0.0);
                    }
                    (generator * Complex64::new(dt, 0.0)).exp()
                })
                .collect();

            let pair_sum: f64 = (0..pairs)
                .map(|pair| {
                    let state_start = pair * dim;
                    let mut state = vector_from_slice(&rho_init[state_start..state_start + dim]);
                    for propagator in &propagators {
                        state = propagator * state;
                    }
                    let target = vector_from_slice(&rho_targ[state_start..state_start + dim]);
                    fidelity_mode.value(target.dotc(&state))
                })
                .sum();
            pair_sum / pairs as f64
        })
        .sum();

    Ok(sum / members as f64)
}

/// Mean fidelity and exact GRAPE gradient for coherent vector-state ensembles.
#[pyfunction]
fn grape_value_gradient_vectors(
    drifts: PyReadonlyArray3<'_, Complex64>,
    operators: PyReadonlyArray4<'_, Complex64>,
    waveform: PyReadonlyArray2<'_, f64>,
    rho_init: PyReadonlyArray2<'_, Complex64>,
    rho_targ: PyReadonlyArray2<'_, Complex64>,
    dt: f64,
    mode: &str,
) -> PyResult<(f64, Vec<Vec<f64>>)> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be finite and positive"));
    }
    let fidelity_mode = FidelityMode::parse(mode)?;
    let drift_shape = drifts.shape().to_vec();
    let operator_shape = operators.shape().to_vec();
    let waveform_shape = waveform.shape().to_vec();
    let init_shape = rho_init.shape().to_vec();
    let target_shape = rho_targ.shape().to_vec();
    let drifts = drifts.as_slice()?;
    let operators = operators.as_slice()?;
    let waveform = waveform.as_slice()?;
    let rho_init = rho_init.as_slice()?;
    let rho_targ = rho_targ.as_slice()?;
    let (members, channels, steps, dim) = validate_shapes(
        drifts,
        &drift_shape,
        operators,
        &operator_shape,
        waveform,
        &waveform_shape,
        rho_init,
        &init_shape,
        rho_targ,
        &target_shape,
    )?;
    let pairs = init_shape[0];
    let matrix_len = dim * dim;

    let member_results: PyResult<Vec<(f64, Vec<f64>)>> = (0..members)
        .into_par_iter()
        .map(|member| {
            let drift_start = member * matrix_len;
            let drift = matrix_from_slice(&drifts[drift_start..drift_start + matrix_len], dim, dim);
            let member_operator_start = member * channels * matrix_len;
            let controls: Vec<DMatrix<Complex64>> = (0..channels)
                .map(|channel| {
                    let start = member_operator_start + channel * matrix_len;
                    matrix_from_slice(&operators[start..start + matrix_len], dim, dim)
                })
                .collect();
            let slice_data: PyResult<Vec<_>> = (0..steps)
                .map(|step| {
                    let mut generator = drift.clone();
                    for channel in 0..channels {
                        generator += &controls[channel]
                            * Complex64::new(waveform[step * channels + channel], 0.0);
                    }
                    coherent_propagator_and_derivatives(&generator, &controls, dt)
                })
                .collect();
            let slice_data = slice_data?;

            let mut value_sum = 0.0;
            let mut gradient = vec![0.0; steps * channels];
            for pair in 0..pairs {
                let state_start = pair * dim;
                let initial = vector_from_slice(&rho_init[state_start..state_start + dim]);
                let target = vector_from_slice(&rho_targ[state_start..state_start + dim]);
                let mut forward = Vec::with_capacity(steps + 1);
                forward.push(initial);
                for (propagator, _) in &slice_data {
                    forward.push(propagator * forward.last().unwrap());
                }
                let mut backward = vec![DVector::zeros(dim); steps + 1];
                backward[steps] = target.clone();
                for step in (0..steps).rev() {
                    backward[step] = slice_data[step].0.adjoint() * &backward[step + 1];
                }

                let overlap = target.dotc(&forward[steps]);
                value_sum += fidelity_mode.value(overlap);
                for step in 0..steps {
                    for channel in 0..channels {
                        let derivative_state = &slice_data[step].1[channel] * &forward[step];
                        let derivative_overlap = backward[step + 1].dotc(&derivative_state);
                        gradient[step * channels + channel] +=
                            fidelity_mode.gradient(overlap, derivative_overlap);
                    }
                }
            }
            let pair_scale = 1.0 / pairs as f64;
            for value in &mut gradient {
                *value *= pair_scale;
            }
            Ok((value_sum * pair_scale, gradient))
        })
        .collect();
    let member_results = member_results?;
    let mut value = 0.0;
    let mut gradient = vec![0.0; steps * channels];
    for (member_value, member_gradient) in member_results {
        value += member_value;
        for (total, contribution) in gradient.iter_mut().zip(member_gradient) {
            *total += contribution;
        }
    }
    let member_scale = 1.0 / members as f64;
    value *= member_scale;
    for item in &mut gradient {
        *item *= member_scale;
    }
    let rows = gradient.chunks(channels).map(|row| row.to_vec()).collect();
    Ok((value, rows))
}

/// Fidelity and exact gradient for every ensemble member and state pair.
///
/// This lower-level kernel supports smooth-min/max robust pulse objectives in
/// Python without repeating matrix decompositions or crossing the Python/Rust
/// boundary for every offset.
#[pyfunction]
fn grape_member_value_gradients_vectors(
    drifts: PyReadonlyArray3<'_, Complex64>,
    operators: PyReadonlyArray4<'_, Complex64>,
    waveform: PyReadonlyArray2<'_, f64>,
    rho_init: PyReadonlyArray2<'_, Complex64>,
    rho_targ: PyReadonlyArray2<'_, Complex64>,
    dt: f64,
    mode: &str,
) -> PyResult<(Vec<Vec<f64>>, MemberGradients)> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be finite and positive"));
    }
    let fidelity_mode = FidelityMode::parse(mode)?;
    let drift_shape = drifts.shape().to_vec();
    let operator_shape = operators.shape().to_vec();
    let waveform_shape = waveform.shape().to_vec();
    let init_shape = rho_init.shape().to_vec();
    let target_shape = rho_targ.shape().to_vec();
    let drifts = drifts.as_slice()?;
    let operators = operators.as_slice()?;
    let waveform = waveform.as_slice()?;
    let rho_init = rho_init.as_slice()?;
    let rho_targ = rho_targ.as_slice()?;
    let (members, channels, steps, dim) = validate_shapes(
        drifts,
        &drift_shape,
        operators,
        &operator_shape,
        waveform,
        &waveform_shape,
        rho_init,
        &init_shape,
        rho_targ,
        &target_shape,
    )?;
    let pairs = init_shape[0];
    let matrix_len = dim * dim;

    let member_results: PyResult<Vec<MemberResult>> = (0..members)
        .into_par_iter()
        .map(|member| {
            let drift_start = member * matrix_len;
            let drift = matrix_from_slice(&drifts[drift_start..drift_start + matrix_len], dim, dim);
            let member_operator_start = member * channels * matrix_len;
            let controls: Vec<DMatrix<Complex64>> = (0..channels)
                .map(|channel| {
                    let start = member_operator_start + channel * matrix_len;
                    matrix_from_slice(&operators[start..start + matrix_len], dim, dim)
                })
                .collect();
            let slice_data: PyResult<Vec<_>> = (0..steps)
                .map(|step| {
                    let mut generator = drift.clone();
                    for channel in 0..channels {
                        generator += &controls[channel]
                            * Complex64::new(waveform[step * channels + channel], 0.0);
                    }
                    coherent_propagator_and_derivatives(&generator, &controls, dt)
                })
                .collect();
            let slice_data = slice_data?;

            let mut pair_values = Vec::with_capacity(pairs);
            let mut pair_gradients = Vec::with_capacity(pairs);
            for pair in 0..pairs {
                let state_start = pair * dim;
                let initial = vector_from_slice(&rho_init[state_start..state_start + dim]);
                let target = vector_from_slice(&rho_targ[state_start..state_start + dim]);
                let mut forward = Vec::with_capacity(steps + 1);
                forward.push(initial);
                for (propagator, _) in &slice_data {
                    forward.push(propagator * forward.last().unwrap());
                }
                let mut backward = vec![DVector::zeros(dim); steps + 1];
                backward[steps] = target.clone();
                for step in (0..steps).rev() {
                    backward[step] = slice_data[step].0.adjoint() * &backward[step + 1];
                }

                let overlap = target.dotc(&forward[steps]);
                pair_values.push(fidelity_mode.value(overlap));
                let mut gradient = vec![0.0; steps * channels];
                for step in 0..steps {
                    for channel in 0..channels {
                        let derivative_state = &slice_data[step].1[channel] * &forward[step];
                        let derivative_overlap = backward[step + 1].dotc(&derivative_state);
                        gradient[step * channels + channel] =
                            fidelity_mode.gradient(overlap, derivative_overlap);
                    }
                }
                pair_gradients.push(gradient.chunks(channels).map(|row| row.to_vec()).collect());
            }
            Ok((pair_values, pair_gradients))
        })
        .collect();
    let member_results = member_results?;
    let (values, gradients) = member_results.into_iter().unzip();
    Ok((values, gradients))
}

fn rotate_bloch(mut state: [f64; 3], field: [f64; 3], dt: f64) -> [f64; 3] {
    let norm = (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt();
    if norm == 0.0 {
        return state;
    }
    let axis = [field[0] / norm, field[1] / norm, field[2] / norm];
    let angle = 2.0 * std::f64::consts::PI * norm * dt;
    let (sin_angle, cos_angle) = angle.sin_cos();
    let dot = axis[0] * state[0] + axis[1] * state[1] + axis[2] * state[2];
    let cross = [
        axis[1] * state[2] - axis[2] * state[1],
        axis[2] * state[0] - axis[0] * state[2],
        axis[0] * state[1] - axis[1] * state[0],
    ];
    for component in 0..3 {
        state[component] = cos_angle * state[component]
            + sin_angle * cross[component]
            + (1.0 - cos_angle) * axis[component] * dot;
    }
    state
}

/// Propagate a three-component Bloch vector over an offset/B1 ensemble.
#[pyfunction]
fn bloch_ensemble(
    initial: PyReadonlyArray1<'_, f64>,
    waveform_xy: PyReadonlyArray2<'_, f64>,
    offsets_hz: PyReadonlyArray1<'_, f64>,
    b1_scales: PyReadonlyArray1<'_, f64>,
    rf_hz: f64,
    dt: f64,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    if !rf_hz.is_finite() || rf_hz < 0.0 {
        return Err(PyValueError::new_err(
            "rf_hz must be finite and non-negative",
        ));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be finite and positive"));
    }
    if initial.shape() != [3] || waveform_xy.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "initial must have shape (3,) and waveform_xy shape (steps, 2)",
        ));
    }
    let initial = initial.as_slice()?;
    let waveform = waveform_xy.as_slice()?;
    let offsets = offsets_hz.as_slice()?;
    let scales = b1_scales.as_slice()?;
    if offsets.is_empty() || scales.is_empty() || waveform.is_empty() {
        return Err(PyValueError::new_err(
            "ensemble and waveform axes must be non-empty",
        ));
    }
    let initial_state = [initial[0], initial[1], initial[2]];
    let points: Vec<[f64; 3]> = (0..scales.len() * offsets.len())
        .into_par_iter()
        .map(|index| {
            let scale = scales[index / offsets.len()];
            let offset = offsets[index % offsets.len()];
            let mut state = initial_state;
            for slice in waveform.chunks_exact(2) {
                let field = [scale * rf_hz * slice[0], scale * rf_hz * slice[1], offset];
                state = rotate_bloch(state, field, dt);
            }
            state
        })
        .collect();
    Ok(points
        .chunks(offsets.len())
        .map(|row| row.iter().map(|state| state.to_vec()).collect())
        .collect())
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(grape_fidelity_vectors, module)?)?;
    module.add_function(wrap_pyfunction!(grape_value_gradient_vectors, module)?)?;
    module.add_function(wrap_pyfunction!(
        grape_member_value_gradients_vectors,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(bloch_ensemble, module)?)?;
    module.add("RUST_ACCELERATOR", true)?;
    Ok(())
}
