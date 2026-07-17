"""Regression tests for trajectory/spectral analysis helpers."""

import numpy as np
import pytest

from optimalcontrol.analysis import spectrogram_data


def test_spectrogram_preserves_dc_power() -> None:
    # Regression: scipy's default detrend='constant' subtracted each segment
    # mean, erasing the carrier/DC power of the RF envelope. A constant pulse
    # must show nonzero power concentrated at 0 Hz.
    waveform = np.zeros((128, 2), dtype=np.float64)
    waveform[:, 0] = 1000.0
    _, freqs, power = spectrogram_data(waveform, (0, 1), dt=1e-5)
    assert power.sum() > 0.0
    # DC is the dominant bin (windowing spreads a little into neighbours).
    per_bin = power.sum(axis=1)
    dc_bin = int(np.argmin(np.abs(freqs)))
    assert dc_bin == int(np.argmax(per_bin))
    assert per_bin[dc_bin] > 0.5 * per_bin.sum()


def test_spectrogram_rejects_non_finite_dt() -> None:
    # Regression: NaN dt slipped past 'dt <= 0.0' and produced all-NaN output.
    waveform = np.zeros((64, 2), dtype=np.float64)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="dt must be positive"):
            spectrogram_data(waveform, (0, 1), dt=bad)
