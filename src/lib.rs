use nalgebra::allocator::Allocator;
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, Dim, DimDiff, DimSub, Dyn, OMatrix, OVector,
    SymmetricEigen, U1,
};
use num_complex::Complex64;
use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

type CMatrix<D> = OMatrix<Complex64, D, D>;
type CVector<D> = OVector<Complex64, D>;
/// Per-slice data: (propagator adjoint, propagator, per-channel derivatives).
type SliceData<D> = (CMatrix<D>, CMatrix<D>, Vec<CMatrix<D>>);
/// Per-member results: (per-pair fidelities, per-pair flat gradients).
type MemberPairResults = (Vec<f64>, Vec<Vec<f64>>);

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

/// Anti-Hermiticity check on a row-major dim x dim slice.
fn is_anti_hermitian_slice(values: &[Complex64], dim: usize) -> bool {
    let scale = values
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    for row in 0..dim {
        for col in 0..dim {
            if (values[row * dim + col] + values[col * dim + row].conj()).norm() > 1.0e-12 * scale
            {
                return false;
            }
        }
    }
    true
}

/// Anti-Hermiticity of a member's drift and all its control operators.
///
/// The step generators are drift + sum(w_k * control_k) with real waveform
/// weights, so checking the drift and controls once covers every time slice.
fn member_is_anti_hermitian(
    drift: &[Complex64],
    operators: &[Complex64],
    dim: usize,
    channels: usize,
) -> bool {
    let matrix_len = dim * dim;
    if !is_anti_hermitian_slice(drift, dim) {
        return false;
    }
    (0..channels).all(|channel| {
        is_anti_hermitian_slice(
            &operators[channel * matrix_len..(channel + 1) * matrix_len],
            dim,
        )
    })
}

/// Statically dispatch a generic kernel on the common small dimensions.
macro_rules! dispatch_dim {
    ($dim:expr, $func:ident, $($args:expr),* $(,)?) => {
        match $dim {
            2 => $func(Const::<2>, $($args),*),
            3 => $func(Const::<3>, $($args),*),
            4 => $func(Const::<4>, $($args),*),
            8 => $func(Const::<8>, $($args),*),
            16 => $func(Const::<16>, $($args),*),
            n => $func(Dyn(n), $($args),*),
        }
    };
}

/// Build i * matrix (Hermitian counterpart of an anti-Hermitian generator).
fn hermitian_from_slice<D: Dim>(dim: D, values: &[Complex64]) -> CMatrix<D>
where
    DefaultAllocator: Allocator<D, D>,
{
    let i = Complex64::new(0.0, 1.0);
    OMatrix::from_row_slice_generic(dim, dim, values) * i
}

/// Accumulate h += w * control without allocating a temporary.
fn add_scaled<D: Dim>(h: &mut CMatrix<D>, control: &CMatrix<D>, weight: f64)
where
    DefaultAllocator: Allocator<D, D>,
{
    for (target, source) in h.iter_mut().zip(control.iter()) {
        *target += source * weight;
    }
}

/// Eigendecomposition of one slice's Hermitian generator plus its propagator:
/// (propagator, eigenvectors, eigenvector adjoint, eigenvalues, phases).
type SliceEigen<D> = (
    CMatrix<D>,
    CMatrix<D>,
    CMatrix<D>,
    OVector<f64, D>,
    Vec<Complex64>,
);

/// Assemble drift_h + sum(w_k * control_h_k) for one slice, eigendecompose it,
/// and build the step propagator V * diag(exp(-i*dt*lambda)) * V^H.
fn slice_propagator<D>(
    drift_h: &CMatrix<D>,
    controls_h: &[CMatrix<D>],
    weights: &[f64],
    minus_i_dt: Complex64,
) -> SliceEigen<D>
where
    D: DimSub<U1>,
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>>,
{
    let mut hermitian = drift_h.clone();
    for (control_h, weight) in controls_h.iter().zip(weights) {
        add_scaled(&mut hermitian, control_h, *weight);
    }
    let decomposition = SymmetricEigen::new(hermitian);
    let vectors = decomposition.eigenvectors;
    let eigenvalues = decomposition.eigenvalues;
    let adjoint = vectors.adjoint();
    let phases: Vec<Complex64> = eigenvalues
        .iter()
        .map(|value| (minus_i_dt * value).exp())
        .collect();
    let mut scaled = vectors.clone();
    for (mut column, phase) in scaled.column_iter_mut().zip(phases.iter()) {
        column *= *phase;
    }
    let propagator = &scaled * &adjoint;
    (propagator, vectors, adjoint, eigenvalues, phases)
}

/// Per-pair fidelities and flat (steps * channels) gradients for one member.
///
/// `drift` and `operators` are row-major slices for this member only and must
/// already be verified anti-Hermitian by the caller.
#[allow(clippy::too_many_arguments)]
fn member_pair_values_gradients<D>(
    dim: D,
    drift: &[Complex64],
    operators: &[Complex64],
    waveform: &[f64],
    rho_init: &[Complex64],
    rho_targ: &[Complex64],
    dt: f64,
    mode: FidelityMode,
    channels: usize,
    steps: usize,
    pairs: usize,
    parallel_steps: bool,
) -> MemberPairResults
where
    D: DimSub<U1>,
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>>,
    CMatrix<D>: Send + Sync,
{
    let n = dim.value();
    let matrix_len = n * n;
    let minus_i_dt = Complex64::new(0.0, -dt);

    let drift_h = hermitian_from_slice(dim, drift);
    let controls_h: Vec<CMatrix<D>> = (0..channels)
        .map(|channel| {
            hermitian_from_slice(
                dim,
                &operators[channel * matrix_len..(channel + 1) * matrix_len],
            )
        })
        .collect();

    let step_slice = |step: usize| -> SliceData<D> {
        let (propagator, vectors, adjoint, eigenvalues, _phases) = slice_propagator(
            &drift_h,
            &controls_h,
            &waveform[step * channels..(step + 1) * channels],
            minus_i_dt,
        );

        let step_derivatives: Vec<CMatrix<D>> = controls_h
            .iter()
            .map(|control_h| {
                let mut weighted = &adjoint * control_h * &vectors;
                for row in 0..n {
                    for col in 0..n {
                        // Daleckii-Krein divided difference in its
                        // cancellation-free sinc form, well-conditioned for
                        // near-degenerate eigenvalue pairs and identical to
                        // the Python expression in grape.py.
                        let half_gap = 0.5 * dt * (eigenvalues[row] - eigenvalues[col]);
                        let sinc = if half_gap == 0.0 {
                            1.0
                        } else {
                            half_gap.sin() / half_gap
                        };
                        let divided = minus_i_dt
                            * (minus_i_dt * (0.5 * (eigenvalues[row] + eigenvalues[col]))).exp()
                            * sinc;
                        weighted[(row, col)] *= divided;
                    }
                }
                &vectors * weighted * &adjoint
            })
            .collect();

        (propagator.adjoint(), propagator, step_derivatives)
    };

    let slice_data: Vec<SliceData<D>> = if parallel_steps {
        // Chunk the slice work: one eigendecomposition is ~1 us, far below
        // rayon's per-task overhead.
        (0..steps)
            .into_par_iter()
            .with_min_len(16)
            .map(step_slice)
            .collect()
    } else {
        (0..steps).map(step_slice).collect()
    };
    let mut propagators: Vec<CMatrix<D>> = Vec::with_capacity(steps);
    let mut adjoints: Vec<CMatrix<D>> = Vec::with_capacity(steps);
    let mut derivatives: Vec<Vec<CMatrix<D>>> = Vec::with_capacity(steps);
    for (adjoint, propagator, step_derivatives) in slice_data {
        adjoints.push(adjoint);
        propagators.push(propagator);
        derivatives.push(step_derivatives);
    }

    let mut pair_values = Vec::with_capacity(pairs);
    let mut pair_gradients = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let state_start = pair * n;
        let initial: CVector<D> = OVector::from_column_slice_generic(
            dim,
            Const::<1>,
            &rho_init[state_start..state_start + n],
        );
        let target: CVector<D> = OVector::from_column_slice_generic(
            dim,
            Const::<1>,
            &rho_targ[state_start..state_start + n],
        );
        let mut forward: Vec<CVector<D>> = Vec::with_capacity(steps + 1);
        forward.push(initial);
        for propagator in &propagators {
            forward.push(propagator * forward.last().unwrap());
        }
        let mut backward: Vec<CVector<D>> = Vec::with_capacity(steps + 1);
        backward.push(target.clone());
        for step in (0..steps).rev() {
            let next = backward.last().unwrap();
            backward.push(&adjoints[step] * next);
        }
        backward.reverse();

        let overlap = target.dotc(&forward[steps]);
        pair_values.push(mode.value(overlap));
        let mut gradient = vec![0.0; steps * channels];
        for step in 0..steps {
            for channel in 0..channels {
                let derivative_state = &derivatives[step][channel] * &forward[step];
                let derivative_overlap = backward[step + 1].dotc(&derivative_state);
                gradient[step * channels + channel] = mode.gradient(overlap, derivative_overlap);
            }
        }
        pair_gradients.push(gradient);
    }
    (pair_values, pair_gradients)
}

/// Mean fidelity over all state pairs of one member via the eigen propagators.
#[allow(clippy::too_many_arguments)]
fn member_fidelity_eigen<D>(
    dim: D,
    drift: &[Complex64],
    operators: &[Complex64],
    waveform: &[f64],
    rho_init: &[Complex64],
    rho_targ: &[Complex64],
    dt: f64,
    mode: FidelityMode,
    channels: usize,
    steps: usize,
    pairs: usize,
    parallel_steps: bool,
) -> f64
where
    D: DimSub<U1>,
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimDiff<D, U1>>,
    CMatrix<D>: Send + Sync,
{
    let n = dim.value();
    let matrix_len = n * n;
    let minus_i_dt = Complex64::new(0.0, -dt);

    let drift_h = hermitian_from_slice(dim, drift);
    let controls_h: Vec<CMatrix<D>> = (0..channels)
        .map(|channel| {
            hermitian_from_slice(
                dim,
                &operators[channel * matrix_len..(channel + 1) * matrix_len],
            )
        })
        .collect();

    let mut states: Vec<CVector<D>> = (0..pairs)
        .map(|pair| {
            OVector::from_column_slice_generic(
                dim,
                Const::<1>,
                &rho_init[pair * n..pair * n + n],
            )
        })
        .collect();

    let step_propagator = |step: usize| -> CMatrix<D> {
        slice_propagator(
            &drift_h,
            &controls_h,
            &waveform[step * channels..(step + 1) * channels],
            minus_i_dt,
        )
        .0
    };

    if parallel_steps {
        let propagators: Vec<CMatrix<D>> = (0..steps)
            .into_par_iter()
            .with_min_len(16)
            .map(step_propagator)
            .collect();
        for propagator in &propagators {
            for state in &mut states {
                *state = propagator * &*state;
            }
        }
    } else {
        for step in 0..steps {
            let propagator = step_propagator(step);
            for state in &mut states {
                *state = &propagator * &*state;
            }
        }
    }

    let pair_sum: f64 = states
        .iter()
        .enumerate()
        .map(|(pair, state)| {
            let target: CVector<D> = OVector::from_column_slice_generic(
                dim,
                Const::<1>,
                &rho_targ[pair * n..pair * n + n],
            );
            mode.value(target.dotc(state))
        })
        .sum();
    pair_sum / pairs as f64
}

/// General (possibly dissipative) fallback via dense matrix exponentials.
#[allow(clippy::too_many_arguments)]
fn member_fidelity_expm(
    drift: &[Complex64],
    operators: &[Complex64],
    waveform: &[f64],
    rho_init: &[Complex64],
    rho_targ: &[Complex64],
    dt: f64,
    mode: FidelityMode,
    dim: usize,
    channels: usize,
    steps: usize,
    pairs: usize,
) -> f64 {
    let matrix_len = dim * dim;
    let drift = DMatrix::from_row_slice(dim, dim, drift);
    let controls: Vec<DMatrix<Complex64>> = (0..channels)
        .map(|channel| {
            DMatrix::from_row_slice(
                dim,
                dim,
                &operators[channel * matrix_len..(channel + 1) * matrix_len],
            )
        })
        .collect();

    let mut states: Vec<DVector<Complex64>> = (0..pairs)
        .map(|pair| DVector::from_column_slice(&rho_init[pair * dim..pair * dim + dim]))
        .collect();
    for step in 0..steps {
        let mut generator = drift.clone();
        for channel in 0..channels {
            add_scaled(
                &mut generator,
                &controls[channel],
                waveform[step * channels + channel],
            );
        }
        let propagator = (generator * Complex64::new(dt, 0.0)).exp();
        for state in &mut states {
            *state = &propagator * &*state;
        }
    }
    let pair_sum: f64 = states
        .iter()
        .enumerate()
        .map(|(pair, state)| {
            let target = DVector::from_column_slice(&rho_targ[pair * dim..pair * dim + dim]);
            mode.value(target.dotc(state))
        })
        .sum();
    pair_sum / pairs as f64
}

fn validate_shapes(
    drift_shape: &[usize],
    operator_shape: &[usize],
    waveform_shape: &[usize],
    init_shape: &[usize],
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
        &drift_shape,
        &operator_shape,
        &waveform_shape,
        &init_shape,
        &target_shape,
    )?;
    let pairs = init_shape[0];
    let matrix_len = dim * dim;
    let parallel_steps = members < rayon::current_num_threads() && members * steps >= 256;

    let sum: f64 = (0..members)
        .into_par_iter()
        .map(|member| {
            let drift = &drifts[member * matrix_len..(member + 1) * matrix_len];
            let member_operators = &operators
                [member * channels * matrix_len..(member + 1) * channels * matrix_len];
            if member_is_anti_hermitian(drift, member_operators, dim, channels) {
                dispatch_dim!(
                    dim,
                    member_fidelity_eigen,
                    drift,
                    member_operators,
                    waveform,
                    rho_init,
                    rho_targ,
                    dt,
                    fidelity_mode,
                    channels,
                    steps,
                    pairs,
                    parallel_steps,
                )
            } else {
                member_fidelity_expm(
                    drift,
                    member_operators,
                    waveform,
                    rho_init,
                    rho_targ,
                    dt,
                    fidelity_mode,
                    dim,
                    channels,
                    steps,
                    pairs,
                )
            }
        })
        .sum();

    Ok(sum / members as f64)
}

/// Shared parallel driver returning per-pair values and gradients per member.
#[allow(clippy::too_many_arguments)]
fn all_member_pair_results(
    drifts: &[Complex64],
    operators: &[Complex64],
    waveform: &[f64],
    rho_init: &[Complex64],
    rho_targ: &[Complex64],
    dt: f64,
    mode: FidelityMode,
    members: usize,
    channels: usize,
    steps: usize,
    dim: usize,
    pairs: usize,
) -> PyResult<Vec<MemberPairResults>> {
    let matrix_len = dim * dim;
    // With fewer members than worker threads, recover parallelism by
    // decomposing the time slices of each member concurrently instead. Tiny
    // problems stay serial: the fork/join overhead outweighs ~1 us slices.
    let parallel_steps = members < rayon::current_num_threads() && members * steps >= 256;
    (0..members)
        .into_par_iter()
        .map(|member| {
            let drift = &drifts[member * matrix_len..(member + 1) * matrix_len];
            let member_operators = &operators
                [member * channels * matrix_len..(member + 1) * channels * matrix_len];
            if !member_is_anti_hermitian(drift, member_operators, dim, channels) {
                return Err(PyValueError::new_err(
                    "Rust gradient acceleration requires anti-Hermitian generators",
                ));
            }
            Ok(dispatch_dim!(
                dim,
                member_pair_values_gradients,
                drift,
                member_operators,
                waveform,
                rho_init,
                rho_targ,
                dt,
                mode,
                channels,
                steps,
                pairs,
                parallel_steps,
            ))
        })
        .collect()
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
        &drift_shape,
        &operator_shape,
        &waveform_shape,
        &init_shape,
        &target_shape,
    )?;
    let pairs = init_shape[0];

    let member_results = all_member_pair_results(
        drifts,
        operators,
        waveform,
        rho_init,
        rho_targ,
        dt,
        fidelity_mode,
        members,
        channels,
        steps,
        dim,
        pairs,
    )?;

    let mut value = 0.0;
    let mut gradient = vec![0.0; steps * channels];
    for (pair_values, pair_gradients) in &member_results {
        value += pair_values.iter().sum::<f64>();
        for pair_gradient in pair_gradients {
            for (total, contribution) in gradient.iter_mut().zip(pair_gradient) {
                *total += contribution;
            }
        }
    }
    let scale = 1.0 / (members as f64 * pairs as f64);
    value *= scale;
    for item in &mut gradient {
        *item *= scale;
    }
    let rows = gradient.chunks(channels).map(|row| row.to_vec()).collect();
    Ok((value, rows))
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
    module.add_function(wrap_pyfunction!(bloch_ensemble, module)?)?;
    Ok(())
}
