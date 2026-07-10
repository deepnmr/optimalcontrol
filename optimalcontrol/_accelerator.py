"""Optional Rust acceleration for performance-critical propagation paths."""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

from optimalcontrol._types import RealArray

try:
    _rust: Any = import_module("optimalcontrol._rust")
except ImportError:  # pragma: no cover - exercised by source-only installations
    _rust = None

RUST_ACCELERATOR_AVAILABLE = _rust is not None

_KernelInputs = tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    RealArray,
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    float,
    str,
]


def _enabled() -> bool:
    """Return whether the native extension should be used for this process."""
    disabled = os.environ.get("OPTIMALCONTROL_DISABLE_RUST", "").strip().lower()
    return RUST_ACCELERATOR_AVAILABLE and disabled not in {"1", "true", "yes", "on"}


def _metadata_supported(problem: Any, waveform: RealArray) -> bool:
    """Return whether metadata ignored by the native kernels is valid."""
    if not isinstance(problem.basis, str) or not problem.basis:
        return False
    if problem.freeze is None:
        return True
    freeze = np.asarray(problem.freeze)
    return bool(freeze.dtype == np.dtype(np.bool_) and freeze.shape == waveform.shape)


def _vector_inputs(problems: Sequence[Any], wfm: RealArray) -> _KernelInputs | None:
    """Prepare native vector-kernel inputs, or return ``None`` if unsupported."""
    if not _enabled() or not problems:
        return None

    first = problems[0]
    init_states = [np.asarray(state, dtype=np.complex128) for state in first.rho_init]
    target_states = [np.asarray(state, dtype=np.complex128) for state in first.rho_targ]
    if not init_states or any(state.ndim != 1 for state in init_states + target_states):
        return None

    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2 or not np.all(np.isfinite(waveform)):
        return None
    if not waveform.flags.c_contiguous:
        waveform = np.ascontiguousarray(waveform)

    for problem in problems:
        if not _metadata_supported(problem, waveform):
            return None
        try:
            levels = [float(level) for level in problem.pwr_levels]
        except (TypeError, ValueError):
            return None
        if any(not np.isfinite(level) or level < 0.0 for level in levels):
            return None
        if (
            problem.pulse_dt != first.pulse_dt
            or problem.fidelity_mode != first.fidelity_mode
            or len(problem.drifts) != 1
            or len(problem.operators) != waveform.shape[1]
            or len(problem.pwr_levels) != waveform.shape[1]
            or len(problem.rho_init) != len(init_states)
            or len(problem.rho_targ) != len(target_states)
        ):
            return None
        for actual, expected in zip(problem.rho_init, init_states):
            if not np.array_equal(np.asarray(actual), expected):
                return None
        for actual, expected in zip(problem.rho_targ, target_states):
            if not np.array_equal(np.asarray(actual), expected):
                return None

    try:
        drifts = np.ascontiguousarray(
            np.stack([np.asarray(problem.drifts[0]) for problem in problems]),
            dtype=np.complex128,
        )
        operators = np.ascontiguousarray(
            np.stack(
                [
                    np.stack(
                        [
                            float(level) * np.asarray(operator, dtype=np.complex128)
                            for level, operator in zip(problem.pwr_levels, problem.operators)
                        ]
                    )
                    for problem in problems
                ]
            ),
            dtype=np.complex128,
        )
        rho_init = np.ascontiguousarray(np.stack(init_states), dtype=np.complex128)
        rho_targ = np.ascontiguousarray(np.stack(target_states), dtype=np.complex128)
    except (TypeError, ValueError):
        return None
    if not all(np.all(np.isfinite(array)) for array in (drifts, operators, rho_init, rho_targ)):
        return None

    return (
        drifts,
        operators,
        waveform,
        rho_init,
        rho_targ,
        float(first.pulse_dt),
        str(first.fidelity_mode),
    )


def _problem_inputs(problem: Any, wfm: RealArray) -> _KernelInputs | None:
    """Build native vector-kernel inputs directly from an unexpanded problem.

    This mirrors ``cartesian_product_ensemble`` followed by ``_vector_inputs``
    but assembles the stacked member arrays with numpy broadcasting instead of
    materialising one ``ControlProblem`` per ensemble member. Member order is
    (drift, power, offset); the kernel's mean reduction is order-invariant.
    Returns ``None`` for any problem the native kernels do not support, so the
    caller falls back to the expansion path (which also raises the proper
    validation errors for malformed problems).
    """
    if not _enabled():
        return None
    if problem.phase_cycle is not None:
        return None
    if not problem.drifts or not problem.operators or not problem.pwr_levels:
        return None
    if not problem.rho_init or len(problem.rho_init) != len(problem.rho_targ):
        return None
    dt = float(problem.pulse_dt)
    if not np.isfinite(dt) or dt <= 0.0:
        return None

    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2 or not np.all(np.isfinite(waveform)):
        return None
    if not _metadata_supported(problem, waveform):
        return None
    if not waveform.flags.c_contiguous:
        waveform = np.ascontiguousarray(waveform)

    try:
        rho_init = np.stack([np.asarray(state, dtype=np.complex128) for state in problem.rho_init])
        rho_targ = np.stack([np.asarray(state, dtype=np.complex128) for state in problem.rho_targ])
        drifts = np.stack([np.asarray(drift, dtype=np.complex128) for drift in problem.drifts])
        operators = np.stack(
            [np.asarray(operator, dtype=np.complex128) for operator in problem.operators]
        )
    except (TypeError, ValueError):
        return None
    if rho_init.ndim != 2 or rho_targ.shape != rho_init.shape:
        return None
    dim = int(drifts.shape[-1])
    if drifts.ndim != 3 or drifts.shape[-2] != dim:
        return None
    if operators.ndim != 3 or operators.shape[-2:] != (dim, dim):
        return None
    if operators.shape[0] != waveform.shape[1] or rho_init.shape[1] != dim:
        return None

    try:
        levels = np.asarray([float(level) for level in problem.pwr_levels], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(levels)) or np.any(levels < 0.0):
        return None
    power_ensemble = levels.size > 1 and levels.size != operators.shape[0]
    if power_ensemble:
        scaled_operators = levels[:, None, None, None] * operators[None, :, :, :]
    else:
        if levels.size != operators.shape[0]:
            return None
        scaled_operators = (levels[:, None, None] * operators)[None, :, :, :]
    n_power = int(scaled_operators.shape[0])

    if (problem.offsets is None) != (problem.offset_operators is None):
        return None
    if problem.offsets is not None:
        try:
            offsets = np.asarray([float(offset) for offset in problem.offsets], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if offsets.size == 0 or not np.all(np.isfinite(offsets)):
            return None
        if not problem.offset_operators:
            return None
        offset_generator = np.zeros((dim, dim), dtype=np.complex128)
        for operator in problem.offset_operators:
            operator_array = np.asarray(operator, dtype=np.complex128)
            if operator_array.shape != (dim, dim):
                return None
            offset_generator += operator_array
    else:
        offsets = None
        offset_generator = None

    n_drift = int(drifts.shape[0])
    drift_members = np.repeat(drifts, n_power, axis=0)
    if offsets is not None and offset_generator is not None:
        drift_members = (
            drift_members[:, None, :, :]
            + offsets[None, :, None, None] * offset_generator[None, None, :, :]
        ).reshape(-1, dim, dim)
        n_offsets = int(offsets.size)
    else:
        n_offsets = 1
    operator_members = np.repeat(np.tile(scaled_operators, (n_drift, 1, 1, 1)), n_offsets, axis=0)
    if not all(
        np.all(np.isfinite(array))
        for array in (drift_members, operator_members, rho_init, rho_targ)
    ):
        return None

    return (
        np.ascontiguousarray(drift_members, dtype=np.complex128),
        np.ascontiguousarray(operator_members, dtype=np.complex128),
        waveform,
        np.ascontiguousarray(rho_init, dtype=np.complex128),
        np.ascontiguousarray(rho_targ, dtype=np.complex128),
        dt,
        str(problem.fidelity_mode),
    )


def _generators_coherent(problem: Any) -> bool:
    """Return True when every generator is anti-Hermitian (coherent dynamics).

    Cheap pre-gate for the gradient kernel, which rejects dissipative members
    anyway: skipping the ensemble marshalling and kernel round-trip here avoids
    paying that cost on every gradient call of a relaxation problem. The check
    mirrors the Rust ``member_is_anti_hermitian`` tolerance on the raw (not
    power-scaled) matrices; near-threshold disagreement is harmless because the
    kernel still validates each member itself.
    """
    matrices = list(problem.drifts) + list(problem.operators)
    if problem.offset_operators is not None:
        matrices += list(problem.offset_operators)
    for matrix in matrices:
        try:
            array = np.asarray(matrix, dtype=np.complex128)
        except (TypeError, ValueError):
            return False
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.size == 0:
            return False
        deviation = float(np.abs(array + array.conj().T).max())
        scale = max(1.0, float(np.abs(array).max()))
        if not deviation <= 1e-12 * scale:
            return False
    return True


def problem_vector_fidelity(problem: Any, wfm: RealArray) -> float | None:
    """Return Rust fidelity computed straight from an unexpanded problem."""
    inputs = _problem_inputs(problem, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    try:
        return float(_rust.grape_fidelity_vectors(*inputs))
    except ValueError:
        return None


def problem_vector_value_gradient(problem: Any, wfm: RealArray) -> tuple[float, RealArray] | None:
    """Return Rust fidelity/gradient computed straight from an unexpanded problem."""
    if not _enabled() or not _generators_coherent(problem):
        return None
    inputs = _problem_inputs(problem, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    try:
        value, gradient = _rust.grape_value_gradient_vectors(*inputs)
    except ValueError:
        return None
    return float(value), np.asarray(gradient, dtype=np.float64)


def vector_fidelity(problems: Sequence[Any], wfm: RealArray) -> float | None:
    """Return Rust-accelerated fidelity, or ``None`` for an unsupported problem."""
    inputs = _vector_inputs(problems, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    try:
        return float(_rust.grape_fidelity_vectors(*inputs))
    except ValueError:
        return None


def vector_value_gradient(
    problems: Sequence[Any], wfm: RealArray
) -> tuple[float, RealArray] | None:
    """Return Rust-accelerated coherent fidelity/gradient when supported."""
    if not _enabled() or not all(_generators_coherent(problem) for problem in problems):
        return None
    inputs = _vector_inputs(problems, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    try:
        value, gradient = _rust.grape_value_gradient_vectors(*inputs)
    except ValueError:
        return None
    return float(value), np.asarray(gradient, dtype=np.float64)
