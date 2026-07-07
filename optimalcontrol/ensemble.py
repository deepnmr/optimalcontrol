"""Ensemble expansion helpers for GRAPE control problems."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TypeVar

import numpy as np

from optimalcontrol._types import Array, RealArray
from optimalcontrol._validation import validate_finite_floats as _validate_float_values
from optimalcontrol._validation import validate_nonempty as _validate_nonempty
from optimalcontrol._validation import validate_square_matrix as _validate_square_matrix
from optimalcontrol.grape import (
    ControlProblem,
    _has_rf_power_ensemble,
    _validate_phase_cycle,
    _validate_same_drift_dimensions,
    _zero_frozen,
    grape_gradient,
    grape_xy,
    grape_xy_and_gradient,
)
from optimalcontrol.penalties import total_penalty

ProblemT = TypeVar("ProblemT")
ResultT = TypeVar("ResultT")


def serial_backend(fn: Callable[[ProblemT], ResultT], problems: list[ProblemT]) -> list[ResultT]:
    """Apply ``fn`` to each problem sequentially, preserving input order."""
    return [fn(problem) for problem in problems]


def joblib_backend(
    fn: Callable[[ProblemT], ResultT],
    problems: list[ProblemT],
    n_jobs: int = -1,
) -> list[ResultT]:
    """Apply ``fn`` to problems with joblib when available, else serially."""
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return serial_backend(fn, problems)

    results = Parallel(n_jobs=n_jobs)(delayed(fn)(problem) for problem in problems)
    return list(results)


def _copy_complex_matrix(matrix: Array) -> Array:
    """Return a complex128 copy of an array-like matrix."""
    return np.asarray(matrix, dtype=np.complex128).copy()


def _copy_complex_states(states: Sequence[Array]) -> list[Array]:
    """Return complex128 copies of state arrays."""
    return [np.asarray(state, dtype=np.complex128).copy() for state in states]


def _validate_power_levels(pwr_levels: list[float]) -> None:
    """Raise ValueError if RF ensemble power levels are invalid."""
    _validate_nonempty("pwr_levels", pwr_levels)
    _validate_float_values("pwr_levels", pwr_levels)
    for index, level in enumerate(pwr_levels):
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


def expand_offsets(cp: ControlProblem) -> list[ControlProblem]:
    """Return one control problem for each configured offset value.

    Each offset member adds ``offset * sum(offset_operators)`` to every drift
    generator in the problem. The expanded members clear the optional offset
    metadata so they can be evaluated directly by the single-drift GRAPE path
    after the drift axis has also been expanded.
    """
    if cp.offsets is None and cp.offset_operators is None:
        return [replace(cp, drifts=[_copy_complex_matrix(drift) for drift in cp.drifts])]
    if cp.offsets is None:
        raise ValueError("offsets must be provided when offset_operators are set")
    if cp.offset_operators is None:
        raise ValueError("offset_operators must be provided when offsets are set")

    _validate_nonempty("offsets", cp.offsets)
    _validate_float_values("offsets", cp.offsets)
    _validate_nonempty("offset_operators", cp.offset_operators)
    generator_dim = _validate_same_drift_dimensions(cp.drifts)

    offset_generator = np.zeros((generator_dim, generator_dim), dtype=np.complex128)
    for index, operator in enumerate(cp.offset_operators):
        dim = _validate_square_matrix(f"offset_operators[{index}]", operator)
        if dim != generator_dim:
            raise ValueError(
                f"offset_operators[{index}] dimension {dim} does not match {generator_dim}"
            )
        offset_generator += np.asarray(operator, dtype=np.complex128)

    expanded: list[ControlProblem] = []
    for offset in cp.offsets:
        offset_drifts = [
            np.asarray(
                _copy_complex_matrix(drift) + np.complex128(offset) * offset_generator,
                dtype=np.complex128,
            )
            for drift in cp.drifts
        ]
        expanded.append(
            replace(
                cp,
                drifts=offset_drifts,
                offsets=None,
                offset_operators=None,
            )
        )
    return expanded


def _phase_cycle_rows(phase_cycle: RealArray | None) -> list[RealArray]:
    """Return phase-cycle rows in radians."""
    if phase_cycle is None:
        return []

    _validate_phase_cycle(phase_cycle)
    array = np.asarray(phase_cycle, dtype=np.float64)
    if array.ndim == 1:
        return [np.asarray([phase], dtype=np.float64) for phase in array]
    return [np.asarray(row, dtype=np.float64).copy() for row in array]


def _phase_factors(row: RealArray, n_states: int) -> Array:
    """Return per-state complex phase factors for a phase-cycle row."""
    if row.size == 1:
        scalar = np.asarray(
            [np.exp(np.complex128(1j) * np.complex128(row[0]))] * n_states,
            dtype=np.complex128,
        )
        return scalar
    if row.size == n_states:
        return np.asarray(np.exp(np.complex128(1j) * row), dtype=np.complex128)
    raise ValueError(
        f"phase_cycle row length {row.size} must be 1 or match rho_init length {n_states}"
    )


def expand_phase_cycle(cp: ControlProblem) -> list[ControlProblem]:
    """Return one problem per phase-cycle row with rotated initial states.

    Phase rows are interpreted as radians. A row with one value applies a global
    phase to every initial state; a row with ``len(rho_init)`` values applies
    per-state phases to the matched source states.
    """
    if cp.phase_cycle is None:
        return [replace(cp, rho_init=_copy_complex_states(cp.rho_init))]

    _validate_nonempty("rho_init", cp.rho_init)
    rows = _phase_cycle_rows(cp.phase_cycle)
    expanded: list[ControlProblem] = []
    for row in rows:
        factors = _phase_factors(row, len(cp.rho_init))
        rho_init = [
            np.asarray(factor * np.asarray(state, dtype=np.complex128), dtype=np.complex128)
            for factor, state in zip(factors, cp.rho_init)
        ]
        expanded.append(replace(cp, rho_init=rho_init, phase_cycle=None))
    return expanded


def _expand_optional_offsets(problems: list[ControlProblem]) -> list[ControlProblem]:
    """Expand the optional offset axis on every problem."""
    expanded: list[ControlProblem] = []
    for problem in problems:
        if problem.offsets is None and problem.offset_operators is None:
            expanded.append(problem)
        else:
            expanded.extend(expand_offsets(problem))
    return expanded


def _expand_optional_phase_cycle(problems: list[ControlProblem]) -> list[ControlProblem]:
    """Expand the optional phase-cycle axis on every problem."""
    expanded: list[ControlProblem] = []
    for problem in problems:
        if problem.phase_cycle is None:
            expanded.append(problem)
        else:
            expanded.extend(expand_phase_cycle(problem))
    return expanded


def cartesian_product_ensemble(cp: ControlProblem) -> list[ControlProblem]:
    """Return the full Cartesian product over active ensemble dimensions."""
    problems: list[ControlProblem] = []
    for drift_problem in expand_drifts(cp):
        if _has_rf_power_ensemble(drift_problem):
            problems.extend(expand_power_levels(drift_problem))
        else:
            problems.append(drift_problem)
    problems = _expand_optional_offsets(problems)
    problems = _expand_optional_phase_cycle(problems)
    return problems


def correlated_rho_match(cp: ControlProblem) -> list[ControlProblem]:
    """Return one problem per matched source-target state pair."""
    _validate_nonempty("rho_init", cp.rho_init)
    _validate_nonempty("rho_targ", cp.rho_targ)
    if len(cp.rho_init) != len(cp.rho_targ):
        raise ValueError("rho_init and rho_targ must contain the same number of states")

    return [
        replace(
            cp,
            rho_init=[np.asarray(rho_init, dtype=np.complex128).copy()],
            rho_targ=[np.asarray(rho_targ, dtype=np.complex128).copy()],
        )
        for rho_init, rho_targ in zip(cp.rho_init, cp.rho_targ)
    ]


def correlated_rho_drift(cp: ControlProblem) -> list[ControlProblem]:
    """Return one problem per matched drift and source-target state pair."""
    _validate_nonempty("drifts", cp.drifts)
    _validate_nonempty("rho_init", cp.rho_init)
    _validate_nonempty("rho_targ", cp.rho_targ)
    if len(cp.rho_init) != len(cp.rho_targ):
        raise ValueError("rho_init and rho_targ must contain the same number of states")
    if len(cp.drifts) != len(cp.rho_init):
        raise ValueError("rho_drift mode requires one drift per source-target pair")

    return [
        replace(
            cp,
            drifts=[_copy_complex_matrix(drift)],
            rho_init=[np.asarray(rho_init, dtype=np.complex128).copy()],
            rho_targ=[np.asarray(rho_targ, dtype=np.complex128).copy()],
        )
        for drift, rho_init, rho_targ in zip(cp.drifts, cp.rho_init, cp.rho_targ)
    ]


def ensemble_fidelity(cp: ControlProblem, wfm: RealArray) -> float:
    """Return mean GRAPE fidelity over Cartesian ensemble members.

    When ``cp.penalties`` is set, the summed penalty value is subtracted once
    from the ensemble mean. Penalties are stripped before member evaluation so
    every acceleration path returns the same penalised value.
    """
    from optimalcontrol._accelerator import problem_vector_fidelity, vector_fidelity

    waveform = np.asarray(wfm, dtype=np.float64)
    if cp.penalties is not None:
        fidelity = ensemble_fidelity(replace(cp, penalties=None), waveform)
        penalty_value, _ = total_penalty(waveform, cp.penalties)
        return fidelity - penalty_value
    direct = problem_vector_fidelity(cp, waveform)
    if direct is not None:
        return direct
    problems = cartesian_product_ensemble(cp)
    accelerated = vector_fidelity(problems, waveform)
    if accelerated is not None:
        return accelerated
    values = [grape_xy(problem, waveform) for problem in problems]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def ensemble_gradient(cp: ControlProblem, wfm: RealArray) -> RealArray:
    """Return mean GRAPE gradient over Cartesian ensemble members.

    When ``cp.penalties`` is set, the penalty gradient is subtracted once from
    the ensemble mean, matching ``ensemble_xy_and_gradient`` on every path.
    """
    waveform = np.asarray(wfm, dtype=np.float64)
    if cp.penalties is not None:
        gradient = ensemble_gradient(replace(cp, penalties=None), waveform)
        _, penalty_gradient = total_penalty(waveform, cp.penalties)
        gradient = np.asarray(gradient - penalty_gradient, dtype=np.float64)
        _zero_frozen(gradient, cp.freeze)
        return gradient
    problems = cartesian_product_ensemble(cp)
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


def ensemble_xy_and_gradient(cp: ControlProblem, wfm: RealArray) -> tuple[float, RealArray]:
    """Return mean GRAPE fidelity and gradient over Cartesian ensemble members.

    When ``cp.penalties`` is set, the penalty value and gradient are applied
    once to the ensemble mean. Penalties are stripped before member evaluation
    so every acceleration path returns the same penalised result.
    """
    from optimalcontrol._accelerator import (
        problem_vector_value_gradient,
        vector_value_gradient,
    )

    waveform = np.asarray(wfm, dtype=np.float64)
    if cp.penalties is not None:
        value, gradient = ensemble_xy_and_gradient(replace(cp, penalties=None), waveform)
        penalty_value, penalty_gradient = total_penalty(waveform, cp.penalties)
        gradient = np.asarray(gradient - penalty_gradient, dtype=np.float64)
        _zero_frozen(gradient, cp.freeze)
        return value - penalty_value, gradient
    accelerated = problem_vector_value_gradient(cp, waveform)
    if accelerated is not None:
        value, gradient = accelerated
        _zero_frozen(gradient, cp.freeze)
        return value, gradient
    problems = cartesian_product_ensemble(cp)
    accelerated = vector_value_gradient(problems, waveform)
    if accelerated is not None:
        value, gradient = accelerated
        _zero_frozen(gradient, cp.freeze)
        return value, gradient
    value_sum = 0.0
    gradient = np.zeros_like(waveform, dtype=np.float64)
    for problem in problems:
        member_value, member_gradient = grape_xy_and_gradient(problem, waveform)
        if member_gradient.shape != gradient.shape:
            raise ValueError(
                f"member gradient shape {member_gradient.shape} does not match "
                f"waveform shape {gradient.shape}"
            )
        value_sum += member_value
        gradient += member_gradient
    member_count = float(len(problems))
    return value_sum / member_count, np.asarray(gradient / member_count, dtype=np.float64)
