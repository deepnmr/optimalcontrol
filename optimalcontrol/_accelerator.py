"""Optional Rust acceleration for performance-critical propagation paths."""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

RealArray = npt.NDArray[np.float64]

try:
    _rust: Any = import_module("optimalcontrol._rust")
except ImportError:  # pragma: no cover - exercised by source-only installations
    _rust = None

RUST_ACCELERATOR_AVAILABLE = _rust is not None


def _enabled() -> bool:
    """Return whether the native extension should be used for this process."""
    disabled = os.environ.get("OPTIMALCONTROL_DISABLE_RUST", "").strip().lower()
    return RUST_ACCELERATOR_AVAILABLE and disabled not in {"1", "true", "yes", "on"}


def _vector_inputs(
    problems: Sequence[Any], wfm: RealArray
) -> (
    tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        RealArray,
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        float,
        str,
    ]
    | None
):
    """Prepare native vector-kernel inputs, or return ``None`` if unsupported."""
    if not _enabled() or not problems:
        return None

    first = problems[0]
    init_states = [np.asarray(state, dtype=np.complex128) for state in first.rho_init]
    target_states = [np.asarray(state, dtype=np.complex128) for state in first.rho_targ]
    if not init_states or any(state.ndim != 1 for state in init_states + target_states):
        return None

    waveform = np.asarray(wfm, dtype=np.float64)
    if waveform.ndim != 2 or not waveform.flags.c_contiguous:
        waveform = np.ascontiguousarray(waveform, dtype=np.float64)

    for problem in problems:
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

    return (
        drifts,
        operators,
        waveform,
        rho_init,
        rho_targ,
        float(first.pulse_dt),
        str(first.fidelity_mode),
    )


def vector_fidelity(problems: Sequence[Any], wfm: RealArray) -> float | None:
    """Return Rust-accelerated fidelity, or ``None`` for an unsupported problem."""
    inputs = _vector_inputs(problems, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    return float(_rust.grape_fidelity_vectors(*inputs))


def vector_value_gradient(
    problems: Sequence[Any], wfm: RealArray
) -> tuple[float, RealArray] | None:
    """Return Rust-accelerated coherent fidelity/gradient when supported."""
    inputs = _vector_inputs(problems, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    try:
        value, gradient = _rust.grape_value_gradient_vectors(*inputs)
    except ValueError:
        return None
    return float(value), np.asarray(gradient, dtype=np.float64)


def vector_member_value_gradients(
    problems: Sequence[Any], wfm: RealArray
) -> tuple[RealArray, RealArray] | None:
    """Return per-ensemble/per-state coherent values and gradients.

    Values have shape ``(members, state_pairs)`` and gradients have shape
    ``(members, state_pairs, steps, channels)``.
    """
    inputs = _vector_inputs(problems, wfm)
    if inputs is None:
        return None
    assert _rust is not None
    try:
        values, gradients = _rust.grape_member_value_gradients_vectors(*inputs)
    except ValueError:
        return None
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(gradients, dtype=np.float64),
    )
