"""JCAMP-DX and FAPT waveform import helpers."""

import math
from pathlib import Path

import numpy as np

from optimalcontrol._io_common import (
    Waveform,
    _input_path,
    _source_hash,
    _validate_channels,
)
from optimalcontrol._types import RealArray


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
