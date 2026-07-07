"""Plotting helpers for controls, efficiency curves, and trajectories."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from optimalcontrol.analysis import expectation_values
from optimalcontrol.crop import crop_robustness_sweep
from optimalcontrol.io import Waveform, _xy_channel_indices
from optimalcontrol.rope import inept_max_efficiency, rope_g

try:
    import matplotlib
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.pyplot as plt
except ImportError as _exc:
    raise ImportError("matplotlib is required for optimalcontrol.plotting") from _exc

ComplexArray = npt.NDArray[np.complex128]


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
    n_rows: int = 1,
    n_cols: int = 1,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Return (fig, ax), creating a new figure when ax is None."""
    if ax is not None:
        return _root_figure(ax), ax
    fig, axes = plt.subplots(n_rows, n_cols)
    if isinstance(axes, np.ndarray):
        return fig, axes.flat[0]
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


def plot_ampl_phase(
    wfm: Waveform,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot the amplitude and phase derived from x/y channels versus time.

    Parameters
    ----------
    wfm:
        Waveform with channels labelled ``x`` and ``y``.
    ax:
        Existing axes to draw on.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_index, y_index = _xy_channel_indices(wfm)

    x_values = wfm.data[x_index, :]
    y_values = wfm.data[y_index, :]
    amplitude = np.hypot(x_values, y_values)
    phase_deg = np.degrees(np.arctan2(y_values, x_values))

    if ax is not None:
        fig = _root_figure(ax)
        ax2 = ax.twinx()
        ax.plot(wfm.times, amplitude, color="C0", label="Amplitude")
        ax2.plot(wfm.times, phase_deg, color="C1", linestyle="--", label="Phase")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Amplitude ({wfm.units})", color="C0")
        ax2.set_ylabel("Phase (deg)", color="C1")
        ax.set_title("Amplitude and Phase")
        return fig

    fig, (ax_ampl, ax_phase) = plt.subplots(2, 1, sharex=True)
    ax_ampl.plot(wfm.times, amplitude, color="C0")
    ax_ampl.set_ylabel(f"Amplitude ({wfm.units})")
    ax_ampl.set_title("Amplitude and Phase")
    ax_phase.plot(wfm.times, phase_deg, color="C1")
    ax_phase.set_xlabel("Time (s)")
    ax_phase.set_ylabel("Phase (deg)")
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


def plot_crop_robustness(
    ka_over_J_values: list[float],
    kc_over_ka_values: list[float],
    J_hz: float,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot a 2D colour map of CROP efficiency over ka/J and kc/ka grids.

    Parameters
    ----------
    ka_over_J_values:
        Axis values for the ka/J dimension.
    kc_over_ka_values:
        Axis values for the kc/ka dimension.
    J_hz:
        Scalar coupling in Hz.
    ax:
        Existing axes to draw on.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not ka_over_J_values:
        raise ValueError("ka_over_J_values must be non-empty")
    if not kc_over_ka_values:
        raise ValueError("kc_over_ka_values must be non-empty")
    if J_hz <= 0.0:
        raise ValueError("J_hz must be positive")

    efficiency = crop_robustness_sweep(
        ka_over_J_values=ka_over_J_values,
        kc_over_ka_values=kc_over_ka_values,
        J_hz=J_hz,
    )

    ka_arr = np.asarray(ka_over_J_values, dtype=np.float64)
    kc_arr = np.asarray(kc_over_ka_values, dtype=np.float64)

    fig, axes = _figure_and_axes(ax)
    img = axes.pcolormesh(
        kc_arr,
        ka_arr,
        efficiency,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    fig.colorbar(img, ax=axes, label="CROP efficiency $\\eta$")
    axes.set_xlabel("$k_c / k_a$")
    axes.set_ylabel("$k_a / J$")
    axes.set_title("CROP Robustness")
    return fig


def plot_trajectory(
    trajectory: list[ComplexArray],
    ops: dict[str, ComplexArray],
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot expectation values of operators along a state trajectory.

    Parameters
    ----------
    trajectory:
        List of density matrices or state vectors as returned by
        :func:`optimalcontrol.analysis.state_trajectory`.
    ops:
        Mapping from label string to operator matrix.
    ax:
        Existing axes to draw on.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not trajectory:
        raise ValueError("trajectory must be non-empty")
    if not ops:
        raise ValueError("ops must be non-empty")

    n_steps = len(trajectory)
    indices = np.arange(n_steps, dtype=np.float64)

    fig, axes = _figure_and_axes(ax)
    for name, values in expectation_values(trajectory, ops).items():
        axes.plot(indices, values, label=name)

    axes.set_xlabel("Time step")
    axes.set_ylabel("Expectation value")
    axes.set_title("State Trajectory")
    axes.legend()
    return fig
