"""Trajectory analysis helpers for GRAPE optimised pulses."""

import copy
import itertools

import numpy as np
from scipy.signal import spectrogram

from optimalcontrol._types import Array, RealArray
from optimalcontrol.grape import (
    ControlProblem,
    _has_ensemble_axes,
    _problem_for_basis,
    forward_propagators,
    forward_states,
    grape_xy,
)


def state_trajectory(cp: ControlProblem, wfm: RealArray) -> list[Array]:
    """Return density matrices at each time slice for the first source state.

    Returns a list of length n_steps + 1: [rho_0, rho_1, ..., rho_N].
    For multi-source problems the first source state rho_init[0] is used; for
    ensemble problems the first Cartesian member is used.
    """
    if _has_ensemble_axes(cp):
        from optimalcontrol.ensemble import cartesian_product_ensemble

        cp = cartesian_product_ensemble(cp)[0]
    propagators = forward_propagators(cp, wfm)
    cp = _problem_for_basis(cp)
    return forward_states(cp.rho_init[0], propagators)


def expectation_values(
    trajectory: list[Array],
    ops: dict[str, Array],
) -> dict[str, RealArray]:
    """Return expectation value of each operator at each time step.

    For a density matrix rho, the expectation value of operator O is
    Re(Tr(O @ rho)).  The returned dict maps each operator name to a
    1-D real array of length len(trajectory).

    Parameters
    ----------
    trajectory:
        List of density matrices or state vectors as returned by
        state_trajectory().
    ops:
        Mapping from label to operator matrix.
    """
    if not trajectory:
        raise ValueError("trajectory must be non-empty")

    result: dict[str, RealArray] = {}
    for name, op in ops.items():
        op_arr = np.asarray(op, dtype=np.complex128)
        values: list[float] = []
        for state in trajectory:
            state_arr = np.asarray(state, dtype=np.complex128)
            if state_arr.ndim == 1:
                ev = float(np.real(np.vdot(state_arr, op_arr @ state_arr)))
            else:
                ev = float(np.real(np.trace(op_arr @ state_arr)))
            values.append(ev)
        result[name] = np.array(values, dtype=np.float64)
    return result


def coherence_order_populations(trajectory: list[Array]) -> RealArray:
    """Return coherence-order populations at each time step.

    Not implemented for the current basis; raises NotImplementedError.
    A future implementation would require explicit product-operator basis
    metadata to decompose density matrices into coherence orders p = -N..N.
    """
    raise NotImplementedError(
        "coherence_order_populations requires explicit product-operator basis "
        "metadata, which is not currently stored in the trajectory."
    )


def correlation_order_populations(trajectory: list[Array]) -> RealArray:
    """Return correlation-order populations at each time step.

    Not implemented for the current basis; raises NotImplementedError.
    A future implementation would require explicit product-operator basis
    metadata to decompose density matrices into spin-correlation orders.
    """
    raise NotImplementedError(
        "correlation_order_populations requires explicit product-operator basis "
        "metadata, which is not currently stored in the trajectory."
    )


def robustness_histogram(
    cp_template: ControlProblem,
    wfm: RealArray,
    param_grid: dict[str, list[object]],
) -> RealArray:
    """Evaluate fidelity for each combination of parameters in the grid.

    Parameters
    ----------
    cp_template:
        Base ControlProblem to modify for each parameter combination.
    wfm:
        Fixed waveform to evaluate, shaped (n_steps, n_channels).
    param_grid:
        Dict mapping ControlProblem field names to lists of replacement values.
        All lists are combined as a Cartesian product.  For a single parameter
        with N values the output is a 1-D array of length N; for K parameters
        with N_1, ..., N_K values the output has shape (N_1, ..., N_K).

    Returns
    -------
    ndarray of float64 fidelity values shaped by the parameter grid.

    Examples
    --------
    Sweep over two offset values::

        grid = {"offsets": [[-100.0], [0.0], [100.0]]}
        hist = robustness_histogram(cp, wfm, grid)  # shape (3,)
    """
    if not param_grid:
        raise ValueError("param_grid must not be empty")

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    shape = tuple(len(v) for v in value_lists)
    fidelities = np.zeros(shape, dtype=np.float64)

    for idx in itertools.product(*(range(n) for n in shape)):
        cp_mod = copy.copy(cp_template)
        for i, key in enumerate(keys):
            setattr(cp_mod, key, value_lists[i][idx[i]])
        fidelities[idx] = grape_xy(cp_mod, wfm)

    return fidelities


def spectrogram_data(
    wfm: RealArray,
    channel_pair: tuple[int, int],
    dt: float,
) -> tuple[RealArray, RealArray, RealArray]:
    """Return spectrogram of the complex envelope formed by two waveform channels.

    Combines channels as a complex signal  x + i*y, then computes the power
    spectrogram using scipy.signal.spectrogram.

    Parameters
    ----------
    wfm:
        Waveform array shaped (n_steps, n_channels).
    channel_pair:
        Indices (x_channel, y_channel) to form the complex envelope.
    dt:
        Time step in seconds.

    Returns
    -------
    times: 1-D array of segment centre times (seconds).
    freqs: 1-D array of frequency bin centres (Hz).
    power: 2-D array of power spectral density, shape (n_freqs, n_times).
    """
    if wfm.ndim != 2:
        raise ValueError(f"wfm must be 2-D, got shape {wfm.shape}")
    n_steps, n_channels = wfm.shape
    ch_x, ch_y = channel_pair
    if not (0 <= ch_x < n_channels and 0 <= ch_y < n_channels):
        raise ValueError(
            f"channel_pair {channel_pair} out of range for waveform with {n_channels} channels"
        )
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    fs = 1.0 / dt
    signal = wfm[:, ch_x].astype(np.complex128) + 1j * wfm[:, ch_y].astype(np.complex128)

    nperseg = min(n_steps, 32)
    freqs_raw, times_raw, Sxx = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        return_onesided=False,
    )

    freqs: RealArray = np.asarray(freqs_raw, dtype=np.float64)
    times: RealArray = np.asarray(times_raw, dtype=np.float64)
    power: RealArray = np.asarray(np.abs(Sxx), dtype=np.float64)
    return times, freqs, power
