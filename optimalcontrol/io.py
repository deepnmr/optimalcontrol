"""Waveform import/export helpers."""

import csv
import hashlib
import inspect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from optimalcontrol._types import RealArray
from optimalcontrol.grape import (
    ControlProblem,
    _basis_name,
    validate_control_problem,
    validate_waveform,
)
from optimalcontrol.optimizers import OptimResult
from optimalcontrol.penalties import PenaltySpec


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

        times = _as_real("times", self.times, 1)
        data = _as_real("data", self.data, 2)
        if data.shape[0] != len(channels):
            raise ValueError(f"data has {data.shape[0]} channels, expected {len(channels)}")
        if data.shape[1] != times.shape[0]:
            raise ValueError(f"data has {data.shape[1]} time steps, expected {times.shape[0]}")

        problem_hash = str(self.problem_hash)
        if not problem_hash.strip():
            raise ValueError("problem_hash must be a non-empty string")

        self.channels = channels
        self.units = units
        self.times = times
        self.data = data
        self.metadata = _validate_metadata(self.metadata)
        self.problem_hash = problem_hash


def _as_real(name: str, value: object, ndim: int) -> RealArray:
    """Return a finite float64 array with the requested dimensionality."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        wording = "one" if ndim == 1 else "two"
        raise ValueError(f"{name} must be {wording}-dimensional, got shape {array.shape}")
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


def _input_path(path: str | Path) -> Path:
    """Return an existing file path suitable for reading."""
    if isinstance(path, str) and not path.strip():
        raise ValueError("path must be a non-empty string")
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_dir():
        raise ValueError("path must point to a file, not a directory")
    return input_path


def _source_hash(prefix: str, payload: bytes) -> str:
    """Return a deterministic hash for imported waveform files."""
    digest = hashlib.sha256(payload).hexdigest()
    return f"{prefix}:{digest}"


def _strip_inline_comment(line: str) -> str:
    """Remove simple inline comments from tabular text lines."""
    stripped = line
    for marker in ("$$", "#"):
        marker_index = stripped.find(marker)
        if marker_index >= 0:
            stripped = stripped[:marker_index]
    return stripped.strip()


def _split_float_tokens(text: str) -> list[float]:
    """Parse comma/whitespace separated finite floats from a text fragment."""
    normalised = text
    for character in ",;()":
        normalised = normalised.replace(character, " ")

    values: list[float] = []
    for token in normalised.split():
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"could not parse float token {token!r}") from exc
        if not math.isfinite(value):
            raise ValueError("numeric table entries must be finite")
        values.append(value)
    return values


def _split_label_list(text: str) -> list[str]:
    """Parse comma/whitespace separated labels from a JCAMP-style value."""
    normalised = text.strip().strip("()")
    if "," in normalised:
        parts = normalised.split(",")
    else:
        parts = normalised.split()
    labels = [part.strip().strip("<>\"'") for part in parts]
    return _validate_channels([label for label in labels if label])


def _numeric_table(lines: list[str], source_name: str) -> RealArray:
    """Parse a rectangular finite numeric table."""
    rows: list[list[float]] = []
    width: int | None = None
    for line_number, line in enumerate(lines, start=1):
        cleaned = _strip_inline_comment(line)
        if not cleaned:
            continue
        values = _split_float_tokens(cleaned)
        if not values:
            continue
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(
                f"{source_name} row {line_number} has {len(values)} columns, expected {width}"
            )
        rows.append(values)

    if not rows:
        raise ValueError(f"{source_name} contains no numeric data")
    return np.asarray(rows, dtype=np.float64)


def _xy_channel_indices(wfm: Waveform) -> tuple[int, int]:
    """Return row indices for x/y waveform channels."""
    lowered = [channel.lower() for channel in wfm.channels]
    if "x" not in lowered or "y" not in lowered:
        raise ValueError("waveform must contain 'x' and 'y' channels")
    return lowered.index("x"), lowered.index("y")


def _bruker_amplitude_phase_deg(wfm: Waveform) -> tuple[RealArray, RealArray]:
    """Return amplitude and degree phase arrays for minimal Bruker export."""
    if len(wfm.channels) == 1:
        amplitude = np.asarray(wfm.data[0, :], dtype=np.float64)
        phase_deg = np.zeros_like(amplitude, dtype=np.float64)
        return amplitude, phase_deg

    if len(wfm.channels) != 2:
        raise ValueError(
            f"Bruker amplitude/phase export supports 1 or 2 (x/y) channels, got {len(wfm.channels)}"
        )
    x_index, y_index = _xy_channel_indices(wfm)
    x_values = np.asarray(wfm.data[x_index, :], dtype=np.float64)
    y_values = np.asarray(wfm.data[y_index, :], dtype=np.float64)
    amplitude = np.asarray(np.hypot(x_values, y_values), dtype=np.float64)
    phase_deg = np.asarray(np.degrees(np.arctan2(y_values, x_values)), dtype=np.float64)
    return amplitude, phase_deg


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


def export_bruker(wfm: Waveform, path: str | Path) -> None:
    """Write a minimal Bruker-style shape text file.

    This is an interoperability stub, not a production-ready Bruker exporter.
    It writes an amplitude/phase ``##XYPOINTS`` table plus a few metadata tags,
    but it does not perform spectrometer calibration, power normalisation,
    vendor-specific shape parameter validation, or safety checks required before
    loading a pulse on hardware.
    """
    amplitude, phase_deg = _bruker_amplitude_phase_deg(wfm)
    output = _output_path(path)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##TITLE= optimalcontrol minimal Bruker shape\n")
        handle.write("##JCAMP-DX= 5.00\n")
        handle.write("##DATA TYPE= Shape Data\n")
        handle.write("##ORIGIN= optimalcontrol\n")
        handle.write("##OWNER= optimalcontrol\n")
        handle.write("##UNITS= " + wfm.units + "\n")
        handle.write("##NPOINTS= " + str(int(wfm.times.shape[0])) + "\n")
        handle.write("##CHANNELS= amplitude,phase_deg\n")
        handle.write("##$OPTIMALCONTROL_PROBLEM_HASH= " + wfm.problem_hash + "\n")
        handle.write(
            "##$OPTIMALCONTROL_LIMITATIONS= "
            "Minimal stub only; not production-ready for spectrometer use.\n"
        )
        handle.write(
            "##$OPTIMALCONTROL_TIMES= "
            + " ".join(_format_float(float(value)) for value in wfm.times)
            + "\n"
        )
        handle.write("##XYPOINTS= (XY..XY)\n")
        for step_index in range(wfm.times.shape[0]):
            handle.write(
                _format_float(float(amplitude[step_index]))
                + ", "
                + _format_float(float(phase_deg[step_index]))
                + "\n"
            )
        handle.write("##END=\n")


def export_bruker_shape(
    path: str | Path,
    title: str,
    amplitude_percent: RealArray,
    phase_deg: RealArray,
    *,
    minx: float = 0.0,
    maxy: float = 360.0,
    totrot: float = 180.0,
    bwfac: float | None = 0.0,
    integfac: float | None = None,
    shape_mode: int = 1,
    extra_tags: Sequence[str] = (),
) -> Path:
    """Write a Bruker shape-library JCAMP file from amplitude/phase arrays.

    This is the shared writer behind the example scripts; :func:`export_bruker`
    remains the minimal :class:`Waveform` round-trip stub. Amplitude is in
    percent of the calibrated RF maximum and phase in degrees. ``extra_tags``
    are raw ``##...`` lines inserted verbatim before ``##NPOINTS``; passing
    ``bwfac=None`` omits the ``##$SHAPE_BWFAC`` tag, and ``integfac`` defaults
    to the mean amplitude fraction.
    """
    amplitude_percent = np.asarray(amplitude_percent, dtype=np.float64)
    phase_deg = np.asarray(phase_deg, dtype=np.float64)
    if integfac is None:
        integfac = float(np.mean(amplitude_percent)) / 100.0
    lines = [
        f"##TITLE= {title}",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Shape Data",
        "##ORIGIN= optimalcontrol",
        "##OWNER= optimalcontrol",
        f"##MINX= {minx:.6e}",
        "##MAXX= 1.000000e+02",
        "##MINY= 0.000000e+00",
        f"##MAXY= {maxy:.6e}",
        "##$SHAPE_EXMODE= None",
        f"##$SHAPE_TOTROT= {totrot:.6e}",
    ]
    if bwfac is not None:
        lines.append(f"##$SHAPE_BWFAC= {bwfac:.6e}")
    lines.append(f"##$SHAPE_INTEGFAC= {integfac:.9e}")
    lines.append(f"##$SHAPE_MODE= {shape_mode}")
    lines.extend(extra_tags)
    lines.append(f"##NPOINTS= {amplitude_percent.size}")
    lines.append("##XYPOINTS= (XY..XY)")
    for amplitude, phase in zip(amplitude_percent, phase_deg, strict=True):
        lines.append(f"{float(amplitude):.9e}, {float(phase):.9e}")
    lines.append("##END=")
    output = _output_path(path)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output


def _parse_jcamp(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse a small subset of JCAMP-DX tags and the XYPOINTS table."""
    tags: dict[str, str] = {}
    xy_lines: list[str] = []
    in_xy_points = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("$$"):
            continue
        if line.startswith("##"):
            body = line[2:]
            if "=" not in body:
                raise ValueError(f"invalid JCAMP tag line {line!r}")
            key_raw, value = body.split("=", 1)
            key = key_raw.strip().upper()
            value = value.strip()
            # JCAMP-DX: '$$' always starts a comment; strip it from the value
            # (not _strip_inline_comment, which also cuts at '#' and would
            # corrupt legitimate '#'-bearing values such as TITLE).
            comment_index = value.find("$$")
            if comment_index != -1:
                value = value[:comment_index].rstrip()
            if key == "END":
                break
            tags[key] = value
            in_xy_points = key == "XYPOINTS"
            continue
        if in_xy_points:
            xy_lines.append(line)

    if "XYPOINTS" not in tags:
        raise ValueError("JCAMP-DX file is missing ##XYPOINTS")
    return tags, xy_lines


def _resolve_jcamp_layout(
    tags: dict[str, str], table: RealArray, n_points: int
) -> tuple[list[str], RealArray, RealArray]:
    """Return (channels, times, data) resolved from JCAMP tags and the table.

    The only supported layout is the one written by :func:`export_bruker`: a
    ``CHANNELS`` label list, a private ``$OPTIMALCONTROL_TIMES`` time axis, and
    one table column per channel.
    """
    channels_raw = tags.get("CHANNELS")
    if channels_raw is None:
        raise ValueError("JCAMP-DX file is missing ##CHANNELS")
    channels = _split_label_list(channels_raw)
    times_raw = tags.get("$OPTIMALCONTROL_TIMES")
    if times_raw is None:
        raise ValueError("JCAMP-DX file is missing ##$OPTIMALCONTROL_TIMES")
    times = np.asarray(_split_float_tokens(times_raw), dtype=np.float64)
    if times.shape[0] != n_points:
        raise ValueError(
            f"JCAMP-DX private time axis has {times.shape[0]} points, expected {n_points}"
        )
    if table.shape[1] != len(channels):
        raise ValueError(
            f"JCAMP-DX table has {table.shape[1]} columns, expected {len(channels)} channels"
        )
    data = np.asarray(table.T, dtype=np.float64)
    return channels, times, data


def import_jcamp_dx(path: str | Path) -> Waveform:
    """Parse a minimal JCAMP-DX waveform file.

    This is a deliberately small import stub, not a full JCAMP-DX reader. It
    recognises flat ``##KEY= value`` tags, a numeric ``##XYPOINTS`` table, and
    requires ``##CHANNELS`` plus ``##$OPTIMALCONTROL_TIMES`` metadata as
    written by :func:`export_bruker`. It does not support compressed JCAMP
    encodings, multi-block files, vendor-specific Bruker shape semantics, or
    spectrometer validation.
    """
    input_file = _input_path(path)
    payload = input_file.read_bytes()
    tags, xy_lines = _parse_jcamp(payload.decode("utf-8"))
    table = _numeric_table(xy_lines, "JCAMP-DX XYPOINTS")

    n_points = int(table.shape[0])
    expected_points_raw = tags.get("NPOINTS")
    if expected_points_raw is not None and int(float(expected_points_raw)) != n_points:
        raise ValueError(
            f"JCAMP-DX NPOINTS={expected_points_raw} does not match {n_points} parsed rows"
        )

    channels, times, data = _resolve_jcamp_layout(tags, table, n_points)

    problem_hash = (
        tags.get("$OPTIMALCONTROL_PROBLEM_HASH")
        or tags.get("PROBLEM_HASH")
        or _source_hash("jcamp-dx", payload)
    )
    metadata: dict[str, object] = {
        "format": "JCAMP-DX",
        "source_path": str(input_file),
        "tags": tags,
    }
    return Waveform(
        channels=channels,
        units=tags.get("UNITS", "a.u."),
        times=times,
        data=data,
        metadata=metadata,
        problem_hash=problem_hash,
    )


def heterodyne_transform(wfm: Waveform, carrier_hz: float) -> Waveform:
    """Return an x/y waveform shifted by a carrier frequency in Hz.

    The exported Cartesian envelope is treated as ``x(t) + i*y(t)`` and rotated
    by ``exp(i*2*pi*carrier_hz*t)``. Passing a negative carrier applies the
    opposite frequency shift.
    """
    if not math.isfinite(carrier_hz):
        raise ValueError("carrier_hz must be finite")
    x_index, y_index = _xy_channel_indices(wfm)
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


def fapt_import(path: str | Path) -> Waveform:
    """Parse a frequency-amplitude-phase-time tabular waveform file.

    The minimal FAPT format accepted here is a text table with four numeric
    columns ordered as frequency in Hz, amplitude in arbitrary units, phase in
    radians, and time in seconds. A single header row and ``#`` or ``$$`` comments
    are ignored.
    """
    input_file = _input_path(path)
    payload = input_file.read_bytes()
    rows: list[list[float]] = []
    header_skipped = False
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        cleaned = _strip_inline_comment(line)
        if not cleaned:
            continue
        try:
            values = _split_float_tokens(cleaned)
        except ValueError:
            if rows or header_skipped:
                raise ValueError(f"invalid FAPT numeric row {line_number}") from None
            header_skipped = True
            continue
        if len(values) != 4:
            raise ValueError(f"FAPT row {line_number} has {len(values)} columns, expected 4")
        rows.append(values)

    if not rows:
        raise ValueError("FAPT file contains no numeric data")

    table = np.asarray(rows, dtype=np.float64)
    metadata: dict[str, object] = {
        "format": "FAPT",
        "source_path": str(input_file),
        "columns": ["frequency_hz", "amplitude", "phase_rad", "time_s"],
    }
    return Waveform(
        channels=["frequency_hz", "amplitude", "phase_rad"],
        units="Hz/a.u./rad",
        times=np.asarray(table[:, 3], dtype=np.float64),
        data=np.asarray(table[:, :3].T, dtype=np.float64),
        metadata=metadata,
        problem_hash=_source_hash("fapt", payload),
    )
