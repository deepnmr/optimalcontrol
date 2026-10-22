"""Waveform import/export helpers."""

import csv
import json
import math
from pathlib import Path
from typing import cast

import numpy as np

from optimalcontrol._io_bruker import _bruker_amplitude_phase_deg as _bruker_amplitude_phase_deg
from optimalcontrol._io_bruker import export_bruker as export_bruker
from optimalcontrol._io_bruker import export_bruker_shape as export_bruker_shape
from optimalcontrol._io_common import Waveform as Waveform
from optimalcontrol._io_common import _as_real as _as_real
from optimalcontrol._io_common import _default_channels as _default_channels
from optimalcontrol._io_common import _format_float as _format_float
from optimalcontrol._io_common import _input_path as _input_path
from optimalcontrol._io_common import _output_path as _output_path
from optimalcontrol._io_common import _source_hash as _source_hash
from optimalcontrol._io_common import _validate_channels as _validate_channels
from optimalcontrol._io_common import _validate_metadata as _validate_metadata
from optimalcontrol._io_common import _xy_channel_indices as _xy_channel_indices
from optimalcontrol._io_common import xy_channel_indices as xy_channel_indices
from optimalcontrol._io_jcamp import _numeric_table as _numeric_table
from optimalcontrol._io_jcamp import _parse_jcamp as _parse_jcamp
from optimalcontrol._io_jcamp import _resolve_jcamp_layout as _resolve_jcamp_layout
from optimalcontrol._io_jcamp import _split_float_tokens as _split_float_tokens
from optimalcontrol._io_jcamp import _split_label_list as _split_label_list
from optimalcontrol._io_jcamp import _strip_inline_comment as _strip_inline_comment
from optimalcontrol._io_jcamp import fapt_import as fapt_import
from optimalcontrol._io_jcamp import import_jcamp_dx as import_jcamp_dx
from optimalcontrol._problem import _callable_provenance as _callable_provenance
from optimalcontrol._problem import _hash_array as _hash_array
from optimalcontrol._problem import _hash_control_problem as _hash_control_problem
from optimalcontrol._problem import _hash_optional_array as _hash_optional_array
from optimalcontrol._problem import _own_callable_provenance as _own_callable_provenance
from optimalcontrol._problem import _penalty_hash_payload as _penalty_hash_payload
from optimalcontrol._types import RealArray
from optimalcontrol.grape import ControlProblem, validate_control_problem, validate_waveform
from optimalcontrol.optimizers import OptimResult


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


def waveform_from_result(
    cp: ControlProblem,
    wfm: RealArray,
    result: OptimResult,
) -> Waveform:
    """Construct an exportable channel-by-time waveform from an optimizer result."""
    validate_control_problem(cp)
    source_waveform = _as_real("wfm", wfm, 2)
    final_waveform = _as_real("result.wfm_final", result.wfm_final, 2)
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
    payload_raw = json.loads(_input_path(path).read_text(encoding="utf-8"))
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


def heterodyne_transform(wfm: Waveform, carrier_hz: float) -> Waveform:
    """Return an x/y waveform shifted by a carrier frequency in Hz.

    The exported Cartesian envelope is treated as ``x(t) + i*y(t)`` and rotated
    by ``exp(i*2*pi*carrier_hz*t)``. Passing a negative carrier applies the
    opposite frequency shift.
    """
    if not math.isfinite(carrier_hz):
        raise ValueError("carrier_hz must be finite")
    x_index, y_index = xy_channel_indices(wfm)
    phase = np.asarray(2.0 * math.pi * carrier_hz * wfm.times, dtype=np.float64)
    envelope = np.asarray(
        wfm.data[x_index, :] + np.complex128(1j) * wfm.data[y_index, :],
        dtype=np.complex128,
    )
    shifted = np.asarray(envelope * np.exp(np.complex128(1j) * phase), dtype=np.complex128)
    data = wfm.data.copy()
    data[x_index, :] = np.real(shifted)
    data[y_index, :] = np.imag(shifted)

    metadata = dict(wfm.metadata)
    metadata["heterodyne_carrier_hz"] = float(carrier_hz)
    metadata["heterodyne_convention"] = "x+i*y multiplied by exp(i*2*pi*carrier_hz*t)"
    return Waveform(
        channels=list(wfm.channels),
        units=wfm.units,
        times=wfm.times.copy(),
        data=data,
        metadata=metadata,
        problem_hash=wfm.problem_hash,
    )
