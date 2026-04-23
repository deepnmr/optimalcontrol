"""Tests for analytical CROP helper formulas."""

import numpy as np
import numpy.testing as npt

from optimalcontrol.crop import crop_eta, crop_kc0_limit, crop_robustness_sweep
from optimalcontrol.rope import rope_g


def test_crop_kc0_limit_matches_rope_efficiency() -> None:
    eta = crop_eta(ka=60.0, kc=0.0, J_hz=100.0)

    npt.assert_allclose(eta, rope_g(n=0.6), rtol=1e-8)
    npt.assert_allclose(crop_kc0_limit(ka=60.0, J_hz=100.0), rope_g(n=0.6), rtol=1e-8)


def test_crop_near_lossless_limit_approaches_unit_efficiency() -> None:
    eta = crop_eta(ka=100.0, kc=99.999, J_hz=100.0)

    npt.assert_allclose(eta, 1.0, atol=0.01)


def test_crop_efficiency_is_monotonic_with_cross_correlation() -> None:
    ka = 100.0
    J_hz = 100.0
    kc_values = [0.0, 0.3 * ka, 0.6 * ka, 0.9 * ka]

    efficiencies = np.array(
        [crop_eta(ka=ka, kc=kc, J_hz=J_hz) for kc in kc_values],
        dtype=np.float64,
    )

    assert np.all(np.diff(efficiencies) >= 0.0)


def test_crop_eta_regression_values() -> None:
    npt.assert_allclose(
        crop_eta(ka=60.0, kc=45.0, J_hz=100.0),
        0.7015664580244381,
        rtol=1e-6,
    )
    npt.assert_allclose(
        crop_eta(ka=110.0, kc=82.5, J_hz=100.0),
        0.5854918065322852,
        rtol=1e-6,
    )


def test_crop_robustness_sweep_stays_within_physical_bound() -> None:
    sweep = crop_robustness_sweep(
        ka_over_J_values=np.linspace(0.0, 2.0, 5),
        kc_over_ka_values=np.linspace(0.0, 1.0, 5),
        J_hz=100.0,
    )

    assert sweep.shape == (5, 5)
    assert np.all(sweep <= 1.0 + 1e-10)
