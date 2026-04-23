"""Waveform import/export helpers."""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, validate_control_problem, validate_waveform
from optimalcontrol.optimizers import OptimResult

RealArray = npt.NDArray[np.float64]


class _Hasher(Protocol):
    """Small protocol for hashlib objects used by problem hashing."""

    def update(self, data: bytes) -> None:
        """Update the hash state."""


@dataclass
class Waveform:
    """Exportable pulse waveform with channels on rows and time on columns."""

    channels: list[str]
    units: str
    times: RealArray
    data: RealArray
    metadata: dict[str, object]
    problem_hash: str

    def __post_init__(self) -> None:
        """Normalise array dtypes and validate the channel-by-time layout."""
        channels = _validate_channels(self.channels)
        units = str(self.units)
        if not units.strip():
            raise ValueError("units must be a non-empty string")

        times = _as_real_1d("times", self.times)
        data = _as_real_2d("data", self.data)
        if data.shape[0] != len(channels):
            raise ValueError(
                f"data has {data.shape[0]} channels, expected {len(channels)}"
            )
        if data.shape[1] != times.shape[0]:
            raise ValueError(
                f"data has {data.shape[1]} time steps, expected {times.shape[0]}"
            )

        problem_hash = str(self.problem_hash)
        if not problem_hash.strip():
            raise ValueError("problem_hash must be a non-empty string")

        self.channels = channels
        self.units = units
        self.times = times
        self.data = data
        self.metadata = _validate_metadata(self.metadata)
        self.problem_hash = problem_hash


def _as_real_1d(name: str, value: object) -> RealArray:
    """Return a finite float64 one-dimensional array."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    return array.copy()


def _as_real_2d(name: str, value: object) -> RealArray:
    """Return a finite float64 two-dimensional array."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    return array.copy()


def _validate_channels(channels: object) -> list[str]:
    """Return validated channel labels."""
    if not isinstance(channels, list):
        raise ValueError("channels must be a list of strings")
    validated: list[str] = []
    for index, channel in enumerate(channels):
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError(f"channels[{index}] must be a non-empty string")
        validated.append(channel)
    if not validated:
        raise ValueError("channels must be non-empty")
    if len(set(validated)) != len(validated):
        raise ValueError("channels must be unique")
    return validated


def _validate_metadata(metadata: object) -> dict[str, object]:
    """Return a shallow copy of metadata with string keys."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    validated: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        validated[key] = value
    return validated


def _default_channels(n_channels: int) -> list[str]:
    """Return deterministic labels for GRAPE control channels."""
    if n_channels == 2:
        return ["x", "y"]
    return [f"ch{index}" for index in range(n_channels)]


def _format_float(value: float) -> str:
    """Return a round-trip-safe decimal representation."""
    return format(float(value), ".17g")


def _jsonable(value: object) -> object:
    """Return a JSON-compatible form of common scalar/container values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON payload cannot contain non-finite floats")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            converted[key] = _jsonable(item)
        return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _jsonable_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return JSON-compatible metadata."""
    converted = _jsonable(metadata)
    if not isinstance(converted, dict):
        raise TypeError("metadata must serialise to a JSON object")
    return cast(dict[str, object], converted)


def _output_path(path: str | Path) -> Path:
    """Return a file path suitable for writing."""
    if isinstance(path, str) and not path.strip():
        raise ValueError("path must be a non-empty string")
    output = Path(path)
    if output.exists() and output.is_dir():
        raise ValueError("path must point to a file, not a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _payload_value(payload: dict[str, object], key: str) -> object:
    """Return a required JSON object field."""
    if key not in payload:
        raise ValueError(f"waveform JSON is missing {key!r}")
    return payload[key]


def _payload_string(payload: dict[str, object], key: str) -> str:
    """Return a required JSON string field."""
    value = _payload_value(payload, key)
    if not isinstance(value, str):
        raise ValueError(f"waveform JSON field {key!r} must be a string")
    return value


def _hash_array(hasher: _Hasher, name: str, value: object) -> None:
    """Add an array payload to a problem hash."""
    array = np.asarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes())


def _hash_optional_array(hasher: _Hasher, name: str, value: object | None) -> None:
    """Add an optional array payload to a problem hash."""
    if value is None:
        hasher.update(f"{name}:none".encode("utf-8"))
        return
    _hash_array(hasher, name, value)


def _hash_control_problem(cp: ControlProblem) -> str:
    """Return a stable hash of the serialisable control-problem definition."""
    hasher = hashlib.sha256()
    scalar_payload: dict[str, object] = {
        "pulse_dt": float(cp.pulse_dt),
        "pwr_levels": [float(level) for level in cp.pwr_levels],
        "fidelity_mode": cp.fidelity_mode,
        "basis": cp.basis,
        "n_drifts": len(cp.drifts),
        "n_operators": len(cp.operators),
        "n_rho_init": len(cp.rho_init),
        "n_rho_targ": len(cp.rho_targ),
        "offsets": None if cp.offsets is None else [float(value) for value in cp.offsets],
    }
    hasher.update(
        json.dumps(scalar_payload, sort_keys=True, allow_nan=False).encode("utf-8")
    )
    for field_name, arrays in (
        ("drifts", cp.drifts),
        ("operators", cp.operators),
        ("rho_init", cp.rho_init),
        ("rho_targ", cp.rho_targ),
    ):
        for index, array in enumerate(arrays):
            _hash_array(hasher, f"{field_name}[{index}]", array)
    if cp.offset_operators is None:
        hasher.update(b"offset_operators:none")
    else:
        for index, array in enumerate(cp.offset_operators):
            _hash_array(hasher, f"offset_operators[{index}]", array)
    _hash_optional_array(hasher, "freeze", cp.freeze)
    _hash_optional_array(hasher, "phase_cycle", cp.phase_cycle)
    return hasher.hexdigest()


def waveform_from_result(
    cp: ControlProblem,
    wfm: RealArray,
    result: OptimResult,
) -> Waveform:
    """Construct an exportable channel-by-time waveform from an optimizer result."""
    validate_control_problem(cp)
    source_waveform = _as_real_2d("wfm", wfm)
    final_waveform = _as_real_2d("result.wfm_final", result.wfm_final)
    if source_waveform.shape != final_waveform.shape:
        raise ValueError(
            f"wfm shape {source_waveform.shape} must match "
            f"result.wfm_final shape {final_waveform.shape}"
        )

    n_steps = int(final_waveform.shape[0])
    n_channels = int(final_waveform.shape[1])
    validate_waveform(source_waveform, n_channels=len(cp.operators), n_steps=n_steps)
    validate_waveform(final_waveform, n_channels=len(cp.operators), n_steps=n_steps)

    times = np.asarray(np.arange(n_steps, dtype=np.float64) * cp.pulse_dt, dtype=np.float64)
    metadata: dict[str, object] = {
        "pulse_dt": float(cp.pulse_dt),
        "pwr_levels": [float(level) for level in cp.pwr_levels],
        "basis": cp.basis,
        "fidelity_mode": cp.fidelity_mode,
        "fidelity_final": float(result.fidelity_final),
        "n_iter": int(result.n_iter),
        "n_feval": int(result.n_feval),
        "converged": bool(result.converged),
        "reason": result.reason,
        "history": [float(value) for value in result.history],
        "source_wfm_shape": [int(size) for size in source_waveform.shape],
    }
    return Waveform(
        channels=_default_channels(n_channels),
        units="a.u.",
        times=times,
        data=np.asarray(final_waveform.T, dtype=np.float64),
        metadata=metadata,
        problem_hash=_hash_control_problem(cp),
    )


def export_csv(wfm: Waveform, path: str | Path) -> None:
    """Write a waveform as CSV with one time column followed by channel columns."""
    output = _output_path(path)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *wfm.channels])
        for step_index, time_value in enumerate(wfm.times):
            row = [_format_float(float(time_value))]
            row.extend(
                _format_float(float(wfm.data[channel_index, step_index]))
                for channel_index in range(len(wfm.channels))
            )
            writer.writerow(row)


def export_json(wfm: Waveform, path: str | Path) -> None:
    """Write a waveform JSON file containing all dataclass fields."""
    payload: dict[str, object] = {
        "channels": list(wfm.channels),
        "units": wfm.units,
        "times": wfm.times.tolist(),
        "data": wfm.data.tolist(),
        "metadata": _jsonable_metadata(wfm.metadata),
        "problem_hash": wfm.problem_hash,
    }
    output = _output_path(path)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def import_json(path: str | Path) -> Waveform:
    """Load a waveform from the JSON format written by :func:`export_json`."""
    payload_raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise ValueError("waveform JSON payload must be an object")
    payload = cast(dict[str, object], payload_raw)
    metadata = _validate_metadata(_payload_value(payload, "metadata"))
    return Waveform(
        channels=_validate_channels(_payload_value(payload, "channels")),
        units=_payload_string(payload, "units"),
        times=np.asarray(_payload_value(payload, "times"), dtype=np.float64),
        data=np.asarray(_payload_value(payload, "data"), dtype=np.float64),
        metadata=metadata,
        problem_hash=_payload_string(payload, "problem_hash"),
    )
