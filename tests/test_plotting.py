"""Tests for plotting helpers."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402

from optimalcontrol.io import Waveform  # noqa: E402
from optimalcontrol.plotting import plot_rope_efficiency, plot_xy_controls  # noqa: E402


def _xy_waveform() -> Waveform:
    return Waveform(
        channels=["x", "y"],
        units="a.u.",
        times=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        data=np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float64),
        metadata={},
        problem_hash="abc",
    )


def test_plot_xy_controls_returns_figure() -> None:
    wfm = _xy_waveform()
    fig = plot_xy_controls(wfm)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_xy_controls_missing_channel_raises() -> None:
    wfm = Waveform(
        channels=["x"],
        units="a.u.",
        times=np.array([0.0], dtype=np.float64),
        data=np.array([[1.0]], dtype=np.float64),
        metadata={},
        problem_hash="abc",
    )
    with pytest.raises(ValueError, match="'x' and 'y'"):
        plot_xy_controls(wfm)


def test_plot_rope_efficiency_returns_figure() -> None:
    fig = plot_rope_efficiency([0.0, 0.5, 1.0], J_hz=100.0)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_rope_efficiency_empty_n_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_rope_efficiency([], J_hz=100.0)
