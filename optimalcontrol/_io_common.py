"""Shared waveform container and path/format helpers for the io package."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optimalcontrol._types import RealArray


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


def xy_channel_indices(wfm: Waveform) -> tuple[int, int]:
    """Return row indices for x/y waveform channels."""
    lowered = [channel.lower() for channel in wfm.channels]
    if "x" not in lowered or "y" not in lowered:
        raise ValueError("waveform must contain 'x' and 'y' channels")
    return lowered.index("x"), lowered.index("y")


_xy_channel_indices = xy_channel_indices
