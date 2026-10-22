"""Control-problem definitions, validation, and stable hashing."""

import hashlib
import inspect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import numpy.typing as npt

from optimalcontrol._types import Array, RealArray
from optimalcontrol._validation import validate_finite_floats as _validate_float_list
from optimalcontrol._validation import validate_nonempty as _validate_nonempty
from optimalcontrol._validation import validate_square_matrix as _validate_square_matrix
from optimalcontrol.penalties import PenaltyInput, PenaltySpec

VALID_FIDELITY_MODES = {"real", "imag", "abs2"}
VALID_BASES = {"dense", "hilbert", "liouville"}


def _raise_invalid_mode(label: str, valid_values: set[str] = VALID_FIDELITY_MODES) -> NoReturn:
    """Raise a ValueError naming the accepted values for ``label``."""
    valid = ", ".join(sorted(valid_values))
    raise ValueError(f"{label} must be one of: {valid}")


def _basis_name(basis: object) -> str:
    """Return a normalised basis name or raise ValueError."""
    if not isinstance(basis, str) or not basis:
        raise ValueError("basis must be a non-empty string")
    normalised = basis.lower()
    if normalised not in VALID_BASES:
        _raise_invalid_mode("basis", VALID_BASES)
    return normalised


@dataclass
class ControlProblem:
    """Spinach-style GRAPE optimisation problem description.

    The drift and control operators define the generator basis used by later
    GRAPE propagation helpers. States may be represented as vectors or square
    matrices; validation accepts both Hilbert-space and vectorised Liouville
    dimensions.
    """

    drifts: list[Array]
    operators: list[Array]
    rho_init: list[Array]
    rho_targ: list[Array]
    pulse_dt: float
    pwr_levels: list[float]
    freeze: npt.NDArray[np.bool_] | None = None
    fidelity_mode: str = "real"
    offsets: list[float] | None = None
    offset_operators: list[Array] | None = None
    phase_cycle: npt.NDArray[np.float64] | None = None
    basis: str = "dense"
    penalties: list[PenaltyInput] | None = None
    checkpoint_path: str | None = None


def _has_rf_power_ensemble(cp: ControlProblem) -> bool:
    """Return True when pwr_levels represents an RF ensemble axis."""
    return len(cp.pwr_levels) > 1 and len(cp.pwr_levels) != len(cp.operators)


def _has_ensemble_axes(cp: ControlProblem) -> bool:
    """Return True when a control problem needs ensemble expansion."""
    return (
        len(cp.drifts) > 1
        or _has_rf_power_ensemble(cp)
        or cp.offsets is not None
        or cp.offset_operators is not None
        or cp.phase_cycle is not None
    )


def _validate_state_shape(name: str, state: Array, generator_dim: int) -> Array:
    """Return a finite state that can be acted on by the generators."""
    array = np.asarray(state, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    if array.ndim == 1:
        if array.shape[0] != generator_dim:
            raise ValueError(
                f"{name} vector length {array.shape[0]} does not match "
                f"generator dimension {generator_dim}"
            )
        return array

    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        matrix_dim = int(array.shape[0])
        liouville_dim = int(array.size)
        if matrix_dim == generator_dim or liouville_dim == generator_dim:
            return array
        raise ValueError(
            f"{name} matrix shape {array.shape} is incompatible with "
            f"generator dimension {generator_dim}"
        )

    raise ValueError(f"{name} must be a vector or square matrix, got shape {array.shape}")


def _validate_explicit_state_shape(
    name: str,
    state: Array,
    generator_dim: int,
    basis: str,
) -> None:
    """Validate state shape requirements imposed by an explicit basis."""
    if basis == "hilbert":
        if state.ndim != 1:
            raise ValueError(f"{name} must be a pure-state vector for Hilbert GRAPE")
        if state.shape[0] != generator_dim:
            raise ValueError(
                f"{name} vector length {state.shape[0]} does not match "
                f"Hilbert generator dimension {generator_dim}"
            )
    elif basis == "liouville":
        if state.ndim == 1:
            if state.shape[0] != generator_dim:
                raise ValueError(
                    f"{name} vector length {state.shape[0]} does not match "
                    f"Liouville generator dimension {generator_dim}"
                )
        elif state.ndim == 2 and state.shape[0] == state.shape[1]:
            if state.size != generator_dim:
                raise ValueError(
                    f"{name} matrix shape {state.shape} does not vectorise to "
                    f"Liouville generator dimension {generator_dim}"
                )
        else:
            raise ValueError(f"{name} must be a vector or square density matrix")


def _validate_freeze_mask(
    freeze: npt.NDArray[np.bool_] | None,
    n_channels: int,
) -> None:
    """Validate the optional waveform freeze mask."""
    if freeze is None:
        return
    mask = np.asarray(freeze)
    if mask.dtype != np.dtype(np.bool_):
        raise ValueError("freeze mask must have boolean dtype")
    if mask.ndim != 2:
        raise ValueError(f"freeze mask must be two-dimensional, got shape {mask.shape}")
    if mask.shape[1] != n_channels:
        raise ValueError(f"freeze mask has {mask.shape[1]} channels, expected {n_channels}")


def _validate_phase_cycle(phase_cycle: npt.NDArray[np.float64] | None) -> None:
    """Validate optional phase-cycle weights or phase rows."""
    if phase_cycle is None:
        return
    array = np.asarray(phase_cycle, dtype=np.float64)
    if array.ndim not in (1, 2) or array.size == 0:
        raise ValueError(f"phase_cycle must be a non-empty 1-D or 2-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("phase_cycle entries must be finite")


def _validate_same_drift_dimensions(drifts: Sequence[Array]) -> int:
    """Validate drift generators and return their shared dimension."""
    _validate_nonempty("drifts", drifts)
    generator_dim = _validate_square_matrix("drifts[0]", drifts[0])
    for index, drift in enumerate(drifts[1:], start=1):
        dim = _validate_square_matrix(f"drifts[{index}]", drift)
        if dim != generator_dim:
            raise ValueError(f"drifts[{index}] dimension {dim} does not match {generator_dim}")
    return generator_dim


def _validate_offset_operators(offset_operators: Sequence[Array], generator_dim: int) -> None:
    """Raise ValueError if any offset operator does not match the generator dimension."""
    for index, operator in enumerate(offset_operators):
        dim = _validate_square_matrix(f"offset_operators[{index}]", operator)
        if dim != generator_dim:
            raise ValueError(
                f"offset_operators[{index}] dimension {dim} does not match {generator_dim}"
            )


def validate_control_problem(cp: ControlProblem) -> None:
    """Raise ValueError if a GRAPE control problem is internally inconsistent."""
    _validate_nonempty("drifts", cp.drifts)
    _validate_nonempty("operators", cp.operators)
    _validate_nonempty("rho_init", cp.rho_init)
    _validate_nonempty("rho_targ", cp.rho_targ)

    if cp.fidelity_mode not in VALID_FIDELITY_MODES:
        _raise_invalid_mode("fidelity_mode", VALID_FIDELITY_MODES)
    basis = _basis_name(cp.basis)
    if cp.pulse_dt <= 0.0 or not math.isfinite(cp.pulse_dt):
        raise ValueError("pulse_dt must be finite and positive")
    if len(cp.rho_init) != len(cp.rho_targ):
        raise ValueError("rho_init and rho_targ must contain the same number of states")
    if len(cp.pwr_levels) != len(cp.operators) and not _has_rf_power_ensemble(cp):
        raise ValueError(
            f"pwr_levels length {len(cp.pwr_levels)} must match operator count {len(cp.operators)}"
        )

    _validate_float_list("pwr_levels", cp.pwr_levels)
    if any(level < 0.0 for level in cp.pwr_levels):
        raise ValueError("pwr_levels entries must be non-negative")

    generator_dim = _validate_same_drift_dimensions(cp.drifts)

    for index, operator in enumerate(cp.operators):
        dim = _validate_square_matrix(f"operators[{index}]", operator)
        if dim != generator_dim:
            raise ValueError(f"operators[{index}] dimension {dim} does not match {generator_dim}")

    for index, (rho_init, rho_targ) in enumerate(zip(cp.rho_init, cp.rho_targ, strict=True)):
        init_shape = np.asarray(rho_init).shape
        target_shape = np.asarray(rho_targ).shape
        if init_shape != target_shape:
            raise ValueError(
                f"rho_init[{index}] shape {init_shape} does not match "
                f"rho_targ[{index}] shape {target_shape}"
            )
        init_name = f"rho_init[{index}]"
        target_name = f"rho_targ[{index}]"
        init = _validate_state_shape(init_name, rho_init, generator_dim)
        target = _validate_state_shape(target_name, rho_targ, generator_dim)
        _validate_explicit_state_shape(init_name, init, generator_dim, basis)
        _validate_explicit_state_shape(target_name, target, generator_dim, basis)

    if cp.offsets is not None:
        _validate_float_list("offsets", cp.offsets)
    if cp.offset_operators is not None:
        _validate_offset_operators(cp.offset_operators, generator_dim)

    _validate_freeze_mask(cp.freeze, len(cp.operators))
    _validate_phase_cycle(cp.phase_cycle)


def validate_waveform(wfm: RealArray, n_channels: int, n_steps: int) -> None:
    """Raise ValueError if a waveform is not shaped as time rows by channels."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    array = np.asarray(wfm, dtype=np.float64)
    expected_shape = (n_steps, n_channels)
    if array.ndim != 2 or array.shape != expected_shape:
        raise ValueError(f"waveform must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("waveform entries must be finite")


def _validate_square_propagator(propagator: Array) -> None:
    """Raise ValueError if a propagator is not square."""
    if propagator.ndim != 2 or propagator.shape[0] != propagator.shape[1]:
        raise ValueError(f"propagator must be square, got shape {propagator.shape}")


def _validate_matches_propagator(name: str, matrix: Array, propagator: Array) -> None:
    """Raise ValueError if a derivative matrix does not match the propagator shape."""
    if matrix.shape != propagator.shape:
        raise ValueError(
            f"{name} shape {matrix.shape} must match propagator shape {propagator.shape}"
        )


def _validate_hilbert_state(state: Array, generator_dim: int, name: str) -> Array:
    """Return a pure Hilbert-space vector state or raise ValueError."""
    state_arr = np.asarray(state, dtype=np.complex128)
    _validate_explicit_state_shape(name, state_arr, generator_dim, "hilbert")
    return state_arr.copy()


def _hash_array(hasher: "hashlib._Hash", name: str, value: object) -> None:
    """Add an array payload to a problem hash."""
    array = np.asarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes())


def _hash_optional_array(hasher: "hashlib._Hash", name: str, value: object | None) -> None:
    """Add an optional array payload to a problem hash."""
    if value is None:
        hasher.update(f"{name}:none".encode())
        return
    _hash_array(hasher, name, value)


def _own_callable_provenance(value: object) -> str:
    """Return a provenance tag stored directly on a callable or instance."""
    try:
        provenance = vars(value).get("__optimalcontrol_provenance__")
    except TypeError:
        provenance = None
    descriptor = vars(type(value)).get("__optimalcontrol_provenance__")
    if provenance is None and inspect.ismemberdescriptor(descriptor):
        provenance = getattr(value, "__optimalcontrol_provenance__", None)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            "callable penalties require a non-empty "
            "__optimalcontrol_provenance__ attribute for stable problem hashes"
        )
    return provenance


def _callable_provenance(penalty: object) -> object:
    """Return stable provenance, including bound-method instance identity."""
    function = getattr(penalty, "__func__", None)
    instance = getattr(penalty, "__self__", None)
    if function is not None and instance is not None:
        return {
            "function": _own_callable_provenance(function),
            "instance": _own_callable_provenance(instance),
        }
    return _own_callable_provenance(penalty)


def _penalty_hash_payload(cp: ControlProblem) -> list[dict[str, object]] | None:
    """Return stable penalty provenance for a control-problem hash."""
    if cp.penalties is None:
        return None
    payload: list[dict[str, object]] = []
    for penalty in cp.penalties:
        if isinstance(penalty, PenaltySpec):
            payload.append(
                {
                    "kind": penalty.kind.upper(),
                    "weight": float(penalty.weight),
                    "limit": None if penalty.limit is None else float(penalty.limit),
                }
            )
        else:
            payload.append({"callable_provenance": _callable_provenance(penalty)})
    return payload


def _hash_control_problem(cp: ControlProblem) -> str:
    """Return a stable hash of the serialisable control-problem definition."""
    hasher = hashlib.sha256()
    scalar_payload: dict[str, object] = {
        "pulse_dt": float(cp.pulse_dt),
        "pwr_levels": [float(level) for level in cp.pwr_levels],
        "fidelity_mode": cp.fidelity_mode,
        "basis": _basis_name(cp.basis),
        "penalties": _penalty_hash_payload(cp),
        "n_drifts": len(cp.drifts),
        "n_operators": len(cp.operators),
        "n_rho_init": len(cp.rho_init),
        "n_rho_targ": len(cp.rho_targ),
        "offsets": None if cp.offsets is None else [float(value) for value in cp.offsets],
    }
    hasher.update(json.dumps(scalar_payload, sort_keys=True, allow_nan=False).encode("utf-8"))
    for field_name, arrays in (
        ("drifts", cp.drifts),
        ("operators", cp.operators),
        ("rho_init", cp.rho_init),
        ("rho_targ", cp.rho_targ),
    ):
        for index, array in enumerate(arrays):
            _hash_array(
                hasher,
                f"{field_name}[{index}]",
                np.asarray(array, dtype=np.complex128),
            )
    if cp.offset_operators is None:
        hasher.update(b"offset_operators:none")
    else:
        for index, array in enumerate(cp.offset_operators):
            _hash_array(
                hasher,
                f"offset_operators[{index}]",
                np.asarray(array, dtype=np.complex128),
            )
    _hash_optional_array(
        hasher,
        "freeze",
        None if cp.freeze is None else np.asarray(cp.freeze, dtype=np.bool_),
    )
    _hash_optional_array(
        hasher,
        "phase_cycle",
        None if cp.phase_cycle is None else np.asarray(cp.phase_cycle, dtype=np.float64),
    )
    return hasher.hexdigest()
