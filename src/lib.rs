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

/// Summed (not yet averaged) fidelity over all state pairs of one member
/// via the eigen propagators. The caller applies the 1/(members*pairs) scale
/// exactly as the gradient kernel does, so both kernels agree bitwise.
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
    pair_sum
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
    pair_sum
}

fn ensure_finite_f64(name: &str, values: &[f64]) -> PyResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} entries must be finite")))
    }
}

fn ensure_finite_c64(name: &str, values: &[Complex64]) -> PyResult<()> {
    if values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} entries must be finite")))
    }
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
    ensure_finite_c64("drifts", drifts)?;
    ensure_finite_c64("operators", operators)?;
    ensure_finite_f64("waveform", waveform)?;
    ensure_finite_c64("rho_init", rho_init)?;
    ensure_finite_c64("rho_targ", rho_targ)?;
    let pairs = init_shape[0];
    let matrix_len = dim * dim;
    let parallel_steps = members < rayon::current_num_threads() && members * steps >= 256;

    // Collect per-member values in index order and reduce serially so the
    // fidelity is bitwise reproducible run-to-run; a parallel f64 sum
    // associates in work-stealing order. Matches the gradient kernel's
    // ordered accumulation.
    let member_values: Vec<f64> = (0..members)
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
        .collect();
    let sum: f64 = member_values.iter().sum();
    let scale = 1.0 / (members as f64 * pairs as f64);

    Ok(sum * scale)
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
    ensure_finite_c64("drifts", drifts)?;
    ensure_finite_c64("operators", operators)?;
    ensure_finite_f64("waveform", waveform)?;
    ensure_finite_c64("rho_init", rho_init)?;
    ensure_finite_c64("rho_targ", rho_targ)?;
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

// ---------------------------------------------------------------------------
// Seedless scaled-unitary spin-1/2 kernel (Buchanan et al. 2025, SI Note 2).
// ---------------------------------------------------------------------------

type Spin = nalgebra::Matrix2<Complex64>;

/// Return the deviation-density operator `a*Ix + b*Iy + c*Iz` for Bloch [a,b,c].
fn spin_operator(a: f64, b: f64, c: f64) -> Spin {
    Spin::new(
        Complex64::new(0.5 * c, 0.0),
        Complex64::new(0.5 * a, -0.5 * b),
        Complex64::new(0.5 * a, 0.5 * b),
        Complex64::new(-0.5 * c, 0.0),
    )
}

/// Analytic constant-element propagator V and its phase derivative G = i[V, Iz].
fn spin_propagator(ux: f64, uy: f64, offset: f64, scale: f64, rf: f64, dt: f64) -> (Spin, Spin) {
    let fx = scale * rf * ux;
    let fy = scale * rf * uy;
    let fz = offset;
    let fnorm = (fx * fx + fy * fy + fz * fz).sqrt();
    if fnorm == 0.0 {
        return (Spin::identity(), Spin::zeros());
    }
    let half = std::f64::consts::PI * fnorm * dt; // theta / 2
    let (s, c) = half.sin_cos();
    let (nx, ny, nz) = (fx / fnorm, fy / fnorm, fz / fnorm);
    let v = Spin::new(
        Complex64::new(c, -s * nz),
        Complex64::new(-s * ny, -s * nx),
        Complex64::new(s * ny, -s * nx),
        Complex64::new(c, s * nz),
    );
    let i = Complex64::new(0.0, 1.0);
    let g = Spin::new(
        Complex64::new(0.0, 0.0),
        -i * v[(0, 1)],
        i * v[(1, 0)],
        Complex64::new(0.0, 0.0),
    );
    (v, g)
}

/// Build per-step propagators and derivatives for one ensemble member.
fn spin_slices(
    waveform: &[f64],
    offset: f64,
    scale: f64,
    rf: f64,
    dt: f64,
) -> (Vec<Spin>, Vec<Spin>) {
    let n = waveform.len() / 2;
    let mut vs = Vec::with_capacity(n);
    let mut gs = Vec::with_capacity(n);
    for k in 0..n {
        let (v, g) = spin_propagator(waveform[2 * k], waveform[2 * k + 1], offset, scale, rf, dt);
        vs.push(v);
        gs.push(g);
    }
    (vs, gs)
}

fn spin_forward(vs: &[Spin], rho_init: Spin) -> (Vec<Spin>, Spin) {
    let mut fwd = Vec::with_capacity(vs.len());
    let mut rho = rho_init;
    for v in vs {
        fwd.push(rho);
        rho = v * rho * v.adjoint();
    }
    (fwd, rho)
}

/// State-to-state fidelity and phase gradient for one member (fidelity = t . Rm).
fn member_pair(
    waveform: &[f64],
    offset: f64,
    scale: f64,
    rf: f64,
    dt: f64,
    rho_init: Spin,
    rho_targ: Spin,
) -> (f64, Vec<f64>) {
    let n = waveform.len() / 2;
    let (vs, gs) = spin_slices(waveform, offset, scale, rf, dt);
    let (fwd, rho_final) = spin_forward(&vs, rho_init);
    let fidelity = 2.0 * (rho_targ * rho_final).trace().re;

    let mut grad = vec![0.0f64; n];
    let mut lam = rho_targ;
    for j in (0..n).rev() {
        let vjh = vs[j].adjoint();
        let gjh = gs[j].adjoint();
        let d_rho = gs[j] * fwd[j] * vjh + vs[j] * fwd[j] * gjh;
        grad[j] = 2.0 * (lam * d_rho).trace().re;
        lam = vjh * lam * vs[j];
    }
    (fidelity, grad)
}

/// Per-step (n^2/2) water-hold cost and gradient for one member (Note 2.7).
fn member_suppress(waveform: &[f64], offset: f64, scale: f64, rf: f64, dt: f64) -> (f64, Vec<f64>) {
    let n = waveform.len() / 2;
    let iz = spin_operator(0.0, 0.0, 1.0);
    let (vs, gs) = spin_slices(waveform, offset, scale, rf, dt);
    let (fwd, _) = spin_forward(&vs, iz);

    let mut cost = 0.0f64;
    let mut grad = vec![0.0f64; n];
    let mut rho_after = iz;
    for prefix in 1..=n {
        rho_after = vs[prefix - 1] * rho_after * vs[prefix - 1].adjoint();
        let mz = 2.0 * (iz * rho_after).trace().re;
        cost += (1.0 - mz) / n as f64;
        let mut lam = iz;
        for j in (0..prefix).rev() {
            let vjh = vs[j].adjoint();
            let gjh = gs[j].adjoint();
            let d_rho = gs[j] * fwd[j] * vjh + vs[j] * fwd[j] * gjh;
            grad[j] += -(2.0 * (lam * d_rho).trace().re) / n as f64;
            lam = vjh * lam * vs[j];
        }
    }
    (cost, grad)
}

/// Borrowed (waveform, offsets, scales, n_steps) from validated kernel inputs.
type SeedlessInputs<'a> = (&'a [f64], &'a [f64], &'a [f64], usize);

/// Validate shared Seedless-kernel inputs and return (waveform, offsets, scales, n_steps).
fn seedless_inputs<'a>(
    waveform_xy: &'a PyReadonlyArray2<'a, f64>,
    offsets_hz: &'a PyReadonlyArray1<'a, f64>,
    b1_scales: &'a PyReadonlyArray1<'a, f64>,
    rf_hz: f64,
    dt: f64,
) -> PyResult<SeedlessInputs<'a>> {
    if !rf_hz.is_finite() || rf_hz < 0.0 {
        return Err(PyValueError::new_err("rf_hz must be finite and non-negative"));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err("dt must be finite and positive"));
    }
    if waveform_xy.shape()[1] != 2 {
        return Err(PyValueError::new_err("waveform_xy must have shape (steps, 2)"));
    }
    let waveform = waveform_xy.as_slice()?;
    let offsets = offsets_hz.as_slice()?;
    let scales = b1_scales.as_slice()?;
    if offsets.is_empty() || scales.is_empty() || waveform.is_empty() {
        return Err(PyValueError::new_err("ensemble and waveform axes must be non-empty"));
    }
    let n_steps = waveform.len() / 2;
    Ok((waveform, offsets, scales, n_steps))
}

/// Per-member S2S fidelities and gradients over the offset-major (offset, B1) grid.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn seedless_pair_value_gradient(
    waveform_xy: PyReadonlyArray2<'_, f64>,
    offsets_hz: PyReadonlyArray1<'_, f64>,
    b1_scales: PyReadonlyArray1<'_, f64>,
    rf_hz: f64,
    dt: f64,
    init_bloch: PyReadonlyArray1<'_, f64>,
    targ_bloch: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (waveform, offsets, scales, n_steps) =
        seedless_inputs(&waveform_xy, &offsets_hz, &b1_scales, rf_hz, dt)?;
    if init_bloch.shape() != [3] || targ_bloch.shape() != [3] {
        return Err(PyValueError::new_err("init_bloch and targ_bloch must have shape (3,)"));
    }
    let i = init_bloch.as_slice()?;
    let t = targ_bloch.as_slice()?;
    let rho_init = spin_operator(i[0], i[1], i[2]);
    let rho_targ = spin_operator(t[0], t[1], t[2]);
    let n_b1 = scales.len();
    let members = offsets.len() * n_b1;

    let results: Vec<(f64, Vec<f64>)> = (0..members)
        .into_par_iter()
        .map(|m| {
            let offset = offsets[m / n_b1];
            let scale = scales[m % n_b1];
            member_pair(waveform, offset, scale, rf_hz, dt, rho_init, rho_targ)
        })
        .collect();

    let mut fidelity = Vec::with_capacity(members);
    let mut gradient = Vec::with_capacity(members * n_steps);
    for (f, g) in results {
        fidelity.push(f);
        gradient.extend_from_slice(&g);
    }
    Ok((fidelity, gradient))
}

/// Per-member per-step suppression cost and gradient over the (offset, B1) grid.
#[pyfunction]
fn seedless_suppress_perstep(
    waveform_xy: PyReadonlyArray2<'_, f64>,
    offsets_hz: PyReadonlyArray1<'_, f64>,
    b1_scales: PyReadonlyArray1<'_, f64>,
    rf_hz: f64,
    dt: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let (waveform, offsets, scales, n_steps) =
        seedless_inputs(&waveform_xy, &offsets_hz, &b1_scales, rf_hz, dt)?;
    let n_b1 = scales.len();
    let members = offsets.len() * n_b1;

    let results: Vec<(f64, Vec<f64>)> = (0..members)
        .into_par_iter()
        .map(|m| {
            let offset = offsets[m / n_b1];
            let scale = scales[m % n_b1];
            member_suppress(waveform, offset, scale, rf_hz, dt)
        })
        .collect();

    let mut cost = Vec::with_capacity(members);
    let mut gradient = Vec::with_capacity(members * n_steps);
    for (c, g) in results {
        cost.push(c);
        gradient.extend_from_slice(&g);
    }
    Ok((cost, gradient))
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(grape_fidelity_vectors, module)?)?;
    module.add_function(wrap_pyfunction!(grape_value_gradient_vectors, module)?)?;
    module.add_function(wrap_pyfunction!(bloch_ensemble, module)?)?;
    module.add_function(wrap_pyfunction!(seedless_pair_value_gradient, module)?)?;
    module.add_function(wrap_pyfunction!(seedless_suppress_perstep, module)?)?;
    Ok(())
}
