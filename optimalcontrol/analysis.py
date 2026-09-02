"""Trajectory analysis helpers for GRAPE optimised pulses."""

import numpy as np
from scipy.signal import spectrogram

from optimalcontrol._types import Array, RealArray
from optimalcontrol.grape import (
    ControlProblem,
    _has_ensemble_axes,
    _problem_for_basis,
    forward_propagators,
    forward_states,
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
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    fs = 1.0 / dt
    signal = wfm[:, ch_x].astype(np.complex128) + 1j * wfm[:, ch_y].astype(np.complex128)

    nperseg = min(n_steps, 32)
    freqs_raw, times_raw, Sxx = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        return_onesided=False,
        detrend=False,  # keep the DC/carrier component of the envelope
    )

    freqs: RealArray = np.asarray(freqs_raw, dtype=np.float64)
    times: RealArray = np.asarray(times_raw, dtype=np.float64)
    power: RealArray = np.asarray(np.abs(Sxx), dtype=np.float64)
    return times, freqs, power
