import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.testing as npt
import pytest

from examples.reburp_pulse import TIME_EVOLUTION_OFFSETS, reburp_waveform, time_evolution_profiles
from optimalcontrol.bloch import propagate_bloch_ensemble


@pytest.mark.parametrize("disable_rust", ["0", "1"])
def test_time_evolution_profiles_match_stepwise_propagation(monkeypatch, disable_rust):
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
    n_steps = 16
    rf_time_product = 6.0
    signed = reburp_waveform(n_steps)
    trajectories = time_evolution_profiles(signed, rf_time_product)

    waveform_xy = np.column_stack((signed, np.zeros_like(signed)))
    offset_t = np.asarray(TIME_EVOLUTION_OFFSETS, dtype=np.float64)
    scales = np.array([1.0])
    initials = {
        "Ix": np.array([1.0, 0.0, 0.0]),
        "Iy": np.array([0.0, 1.0, 0.0]),
        "Iz": np.array([0.0, 0.0, 1.0]),
    }
    for name, initial in initials.items():
        trajectory = trajectories[name]
        assert trajectory.shape == (n_steps + 1, offset_t.size, 3)
        npt.assert_array_equal(trajectory[0], np.broadcast_to(initial, trajectory[0].shape))
        for k in range(1, n_steps + 1):
            expected = propagate_bloch_ensemble(
                initial, waveform_xy[:k], offset_t, scales, rf_time_product, 1.0 / n_steps
            )[0]
            assert np.array_equal(trajectory[k], expected), (name, k)
