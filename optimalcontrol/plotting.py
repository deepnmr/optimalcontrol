"""Plotting helpers for controls, efficiency curves, and trajectories."""

from __future__ import annotations

import numpy as np

from optimalcontrol.io import Waveform, _xy_channel_indices
from optimalcontrol.rope import inept_max_efficiency, rope_g

try:
    import matplotlib
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.pyplot as plt
except ImportError as _exc:
    raise ImportError("matplotlib is required for optimalcontrol.plotting") from _exc

def _root_figure(ax: matplotlib.axes.Axes) -> matplotlib.figure.Figure:
    """Return the top-level Figure owning ax; unlike get_figure(root=True), works on mpl < 3.10."""
    fig = ax.get_figure()
    while isinstance(fig, matplotlib.figure.SubFigure):
        fig = fig.figure
    if fig is None:
        raise ValueError("provided ax has no associated figure")
    return fig


def _figure_and_axes(
    ax: matplotlib.axes.Axes | None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Return (fig, ax), creating a new figure when ax is None."""
    if ax is not None:
        return _root_figure(ax), ax
    fig, axes = plt.subplots()
    return fig, axes


def plot_xy_controls(
    wfm: Waveform,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot the x and y control channels of a waveform versus time.

    Parameters
    ----------
    wfm:
        Waveform with at least two channels labelled ``x`` and ``y``.
    ax:
        Existing axes to draw on.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_index, y_index = _xy_channel_indices(wfm)

    fig, axes = _figure_and_axes(ax)
    times = wfm.times
    axes.plot(times, wfm.data[x_index, :], label="$u_x$")
    axes.plot(times, wfm.data[y_index, :], label="$u_y$")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel(f"Amplitude ({wfm.units})")
    axes.set_title("XY Controls")
    axes.legend()
    return fig


def plot_rope_efficiency(
    n_values: list[float],
    J_hz: float,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot ROPE and INEPT efficiency versus the relaxation parameter n.

    Parameters
    ----------
    n_values:
        Sequence of n = k/J values to evaluate.
    J_hz:
        Scalar coupling in Hz (used only to compute INEPT optimal time).
    ax:
        Existing axes to draw on.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not n_values:
        raise ValueError("n_values must be non-empty")
    if J_hz <= 0.0:
        raise ValueError("J_hz must be positive")

    rope_eff = np.array([rope_g(n) for n in n_values], dtype=np.float64)
    inept_eff = np.array(
        [inept_max_efficiency(n, J_hz) for n in n_values], dtype=np.float64
    )
    n_arr = np.asarray(n_values, dtype=np.float64)

    fig, axes = _figure_and_axes(ax)
    axes.plot(n_arr, rope_eff, label="ROPE")
    axes.plot(n_arr, inept_eff, linestyle="--", label="INEPT")
    axes.set_xlabel("$n = k/J$")
    axes.set_ylabel("Transfer efficiency")
    axes.set_title("ROPE vs INEPT Efficiency")
    axes.legend()
    return fig


