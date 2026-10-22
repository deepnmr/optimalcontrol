"""Deterministic PNG output for example scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
