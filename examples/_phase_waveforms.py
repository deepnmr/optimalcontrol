"""Shared phase-to-waveform helper for the phase-only examples."""

import numpy as np
import numpy.typing as npt

RealArray = npt.NDArray[np.float64]


def phase_to_xy(phase_rad: RealArray) -> RealArray:
    """Return a unit-amplitude Cartesian waveform from phase samples."""
    return np.asarray(
        np.column_stack((np.cos(phase_rad), np.sin(phase_rad))),
        dtype=np.float64,
    )
