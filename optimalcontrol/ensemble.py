"""Ensemble expansion helpers for GRAPE control problems."""

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, grape_gradient, grape_xy

Array = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


def _copy_complex_matrix(matrix: Array) -> Array:
    """Return a complex128 copy of an array-like matrix."""
    return np.asarray(matrix, dtype=np.complex128).copy()


def _validate_nonempty(name: str, values: Sequence[object]) -> None:
    """Raise ValueError if an ensemble axis is empty."""
    if not values:
        raise ValueError(f"{name} must be non-empty")


def _validate_power_levels(pwr_levels: list[float]) -> None:
    """Raise ValueError if RF ensemble power levels are invalid."""
    _validate_nonempty("pwr_levels", pwr_levels)
    for index, level in enumerate(pwr_levels):
        if not math.isfinite(level):
            raise ValueError(f"pwr_levels[{index}] must be finite")
        if level < 0.0:
            raise ValueError(f"pwr_levels[{index}] must be non-negative")


def expand_drifts(cp: ControlProblem) -> list[ControlProblem]:
    """Return one control problem for each drift generator in ``cp``."""
    _validate_nonempty("drifts", cp.drifts)
    return [replace(cp, drifts=[_copy_complex_matrix(drift)]) for drift in cp.drifts]


def expand_power_levels(cp: ControlProblem) -> list[ControlProblem]:
    """Return one control problem for each RF power scaling factor.

    The expanded problems absorb each scalar RF scale into every control operator
    and reset per-channel ``pwr_levels`` to one. This keeps the generated
    problems compatible with the existing single-problem GRAPE propagation path.
    """
    _validate_nonempty("operators", cp.operators)
    _validate_power_levels(cp.pwr_levels)

    expanded: list[ControlProblem] = []
    for level in cp.pwr_levels:
        scaled_operators = [
            np.asarray(np.complex128(level) * _copy_complex_matrix(operator), dtype=np.complex128)
            for operator in cp.operators
        ]
        expanded.append(
            replace(
                cp,
                operators=scaled_operators,
                pwr_levels=[1.0] * len(scaled_operators),
            )
        )
    return expanded


def _expand_drift_power_ensemble(cp: ControlProblem) -> list[ControlProblem]:
    """Return the Cartesian product over drift generators and RF power levels."""
    problems: list[ControlProblem] = []
    for drift_problem in expand_drifts(cp):
        problems.extend(expand_power_levels(drift_problem))
    return problems


def ensemble_fidelity(cp: ControlProblem, wfm: RealArray) -> float:
    """Return mean GRAPE fidelity over drift and RF-power ensemble members."""
    waveform = np.asarray(wfm, dtype=np.float64)
    values = [
        grape_xy(problem, waveform)
        for problem in _expand_drift_power_ensemble(cp)
    ]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def ensemble_gradient(cp: ControlProblem, wfm: RealArray) -> RealArray:
    """Return mean GRAPE gradient over drift and RF-power ensemble members."""
    waveform = np.asarray(wfm, dtype=np.float64)
    problems = _expand_drift_power_ensemble(cp)
    gradient = np.zeros_like(waveform, dtype=np.float64)
    for problem in problems:
        member_gradient = np.asarray(grape_gradient(problem, waveform), dtype=np.float64)
        if member_gradient.shape != gradient.shape:
            raise ValueError(
                f"member gradient shape {member_gradient.shape} does not match "
                f"waveform shape {gradient.shape}"
            )
        gradient += member_gradient
    return np.asarray(gradient / float(len(problems)), dtype=np.float64)
