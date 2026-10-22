import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.testing as npt
import pytest

from examples.methyl_water_binary_symmetric_180 import (
    DURATION_S,
    RF_MAX_HZ,
    SPECTROMETER_1H_MHZ,
    TIME_EVOLUTION_PPM,
    WATER_PPM,
    _xy_waveform,
    time_evolution_profiles,
)
from optimalcontrol.bloch import propagate_bloch_ensemble


@pytest.mark.parametrize("disable_rust", ["0", "1"])
def test_time_evolution_profiles_match_stepwise_propagation(monkeypatch, disable_rust):
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
    n_steps = 16
    half = np.random.default_rng(5).uniform(-0.5, 0.5, n_steps // 2)
    signed = np.concatenate((half, half[::-1]))
    trajectories = time_evolution_profiles(
        signed, ppms=TIME_EVOLUTION_PPM, duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ
    )

    waveform_xy = _xy_waveform(signed)
    offsets_hz = np.array(
        [(ppm - WATER_PPM) * SPECTROMETER_1H_MHZ for ppm in TIME_EVOLUTION_PPM],
        dtype=np.float64,
    )
    scales = np.array([1.0])
    dt = DURATION_S / n_steps
    initials = {
        "Ix": np.array([1.0, 0.0, 0.0]),
        "Iy": np.array([0.0, 1.0, 0.0]),
        "Iz": np.array([0.0, 0.0, 1.0]),
    }
    for name, initial in initials.items():
        trajectory = trajectories[name]
        assert trajectory.shape == (n_steps + 1, offsets_hz.size, 3)
        npt.assert_array_equal(trajectory[0], np.broadcast_to(initial, trajectory[0].shape))
        for k in range(1, n_steps + 1):
            expected = propagate_bloch_ensemble(
                initial, waveform_xy[:k], offsets_hz, scales, RF_MAX_HZ, dt
            )[0]
            assert np.array_equal(trajectory[k], expected), (name, k)
