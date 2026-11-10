"""Deterministic PNG output for example scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

RealArray = npt.NDArray[np.float64]

# 12 decimals sits 8 orders below the precision the figures convey and far above
# the ~1e-14 Rust-vs-NumPy Bloch difference, which Agg's antialiaser can otherwise
# turn into single-pixel PNG changes. Only plotted copies are rounded; library
# return values are never rounded.
QUANTISE_DECIMALS: Final = 12


def quantise(values: RealArray) -> RealArray:
    """Round Bloch-derived curves to ``QUANTISE_DECIMALS`` just before plotting."""
    return np.round(np.asarray(values, dtype=np.float64), QUANTISE_DECIMALS)


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 160,
    bbox_inches: str | None = None,
) -> Path:
    """Save ``fig`` to ``path`` without the matplotlib version stamp, then close it."""
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, metadata={"Software": None})
    plt.close(fig)
    return Path(path)
