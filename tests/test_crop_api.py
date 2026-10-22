"""Direct tests for the public API of optimalcontrol.crop.

Each public function is checked against an analytical identity from
docs/equations.md (CROP symbol table) and docs/guide_rope_crop.md, plus
ValueError pins that use the exact messages from optimalcontrol._validation.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from optimalcontrol.crop import (
    CROPPulse,
    crop_eta,
    crop_eta_prime,
    crop_limit_2IzSz_to_Sz,
    crop_limit_Iz_to_2IzSz,
    crop_limit_Iz_to_Sz,
    crop_limit_single_transition,
    crop_lossless_limit,
    crop_pulse_params,
    crop_waveform,
    crop_zeta,
    single_transition_decomposition,
)
from optimalcontrol.spin_system import Spin, SpinSystem, two_spin_system


def _two_spin(ka: float, kc: float) -> SpinSystem:
    return two_spin_system(
        J_hz=100.0,
        kDD=1.0,
        kCSA_I=1.0,
        kCSA_S=1.0,
        ka=ka,
        kc=kc,
        ka_prime=75.0,
        kc_prime=50.0,
    )


def test_crop_pulse_params_returns_residual_bandwidth_and_carrier() -> None:
    params = crop_pulse_params(ka=60.0, kc=45.0, J_hz=100.0)

    assert isinstance(params, CROPPulse)
    npt.assert_allclose(params.amplitude, math.sqrt(60.0**2 - 45.0**2), rtol=1e-8)
    npt.assert_allclose(params.irradiation_freq_hz, -50.0, rtol=1e-8)
    npt.assert_allclose(params.truncation_window, 1.0 / params.amplitude, rtol=1e-8)


def test_crop_pulse_params_is_lossless_limit_without_residual_bandwidth() -> None:
    params = crop_pulse_params(ka=60.0, kc=60.0, J_hz=100.0)

    assert params.amplitude == 0.0
    assert math.isinf(params.truncation_window)


def test_crop_zeta_matches_closed_form_and_vanishes_at_equal_rates() -> None:
    npt.assert_allclose(
        crop_zeta(ka=60.0, kc=45.0, J_hz=100.0),
        math.sqrt((60.0**2 - 45.0**2) / (100.0**2 + 45.0**2)),
        rtol=1e-8,
    )
    npt.assert_allclose(crop_zeta(ka=60.0, kc=0.0, J_hz=100.0), 0.6, rtol=1e-8)
    npt.assert_allclose(crop_zeta(ka=60.0, kc=60.0, J_hz=100.0), 0.0, atol=1e-8)


def test_crop_eta_equals_rope_efficiency_of_zeta() -> None:
    zeta = crop_zeta(ka=60.0, kc=45.0, J_hz=100.0)

    npt.assert_allclose(
        crop_eta(ka=60.0, kc=45.0, J_hz=100.0), math.sqrt(1.0 + zeta**2) - zeta, atol=1e-8
    )


def test_crop_eta_prime_is_crop_efficiency_of_s_rates() -> None:
    eta_prime = crop_eta_prime(ka_prime=75.0, kc_prime=50.0, J_hz=100.0)
    zeta = crop_zeta(ka=75.0, kc=50.0, J_hz=100.0)

    npt.assert_allclose(eta_prime, math.sqrt(1.0 + zeta**2) - zeta, atol=1e-8)
    assert 0.0 <= eta_prime <= 1.0


def test_crop_limit_Iz_to_2IzSz_is_eta() -> None:
    npt.assert_allclose(crop_limit_Iz_to_2IzSz(0.42), 0.42, atol=1e-8)


def test_crop_limit_2IzSz_to_Sz_is_eta_prime() -> None:
    npt.assert_allclose(crop_limit_2IzSz_to_Sz(0.37), 0.37, atol=1e-8)


def test_crop_limit_Iz_to_Sz_is_product_of_two_steps() -> None:
    npt.assert_allclose(crop_limit_Iz_to_Sz(0.6, 0.5), 0.3, atol=1e-8)


def test_crop_limit_single_transition_is_quadrature_sum() -> None:
    npt.assert_allclose(crop_limit_single_transition(0.6, 0.8), 1.0, atol=1e-8)
    npt.assert_allclose(
        crop_limit_single_transition(0.3, 0.4),
        math.hypot(0.3, 0.4),
        atol=1e-8,
    )


def test_crop_lossless_limit_is_unit_efficiency_at_equal_rates() -> None:
    npt.assert_allclose(crop_lossless_limit(ka=0.0, J_hz=100.0), 1.0, atol=1e-8)
    npt.assert_allclose(
        crop_lossless_limit(ka=60.0, J_hz=100.0),
        crop_eta(ka=60.0, kc=60.0, J_hz=100.0),
        atol=1e-8,
    )


def test_crop_eta_approaches_lossless_limit_as_kc_approaches_ka() -> None:
    ka = 60.0
    eta = crop_eta(ka=ka, kc=ka * (1.0 - 1e-14), J_hz=100.0)

    npt.assert_allclose(eta, crop_lossless_limit(ka=ka, J_hz=100.0), atol=1e-6)


def test_single_transition_decomposition_labels_slow_beta_component() -> None:
    labels = single_transition_decomposition(_two_spin(ka=60.0, kc=45.0))

    assert labels == {"slowly_relaxing": "IzSbeta", "fast_relaxing": "IzSalpha"}


def test_single_transition_decomposition_tie_at_zero_kc_uses_beta() -> None:
    labels = single_transition_decomposition(_two_spin(ka=60.0, kc=0.0))

    assert labels == {"slowly_relaxing": "IzSbeta", "fast_relaxing": "IzSalpha"}


def test_crop_waveform_returns_documented_keys_with_equal_finite_arrays() -> None:
    params = crop_pulse_params(ka=60.0, kc=45.0, J_hz=100.0)
    dt = params.truncation_window / 200.0
    waveform = crop_waveform(
        ka=60.0,
        kc=45.0,
        J_hz=100.0,
        dt=dt,
        truncation_window=params.truncation_window,
    )

    assert set(waveform) == {"times", "amplitude", "irrad_freq"}
    n_steps = len(waveform["times"])
    assert n_steps == math.ceil(params.truncation_window / dt)
    for key in ("times", "amplitude", "irrad_freq"):
        assert len(waveform[key]) == n_steps
        assert np.all(np.isfinite(waveform[key]))
    npt.assert_allclose(waveform["amplitude"], params.amplitude, rtol=1e-8)
    npt.assert_allclose(waveform["irrad_freq"], -50.0, rtol=1e-8)


def test_crop_waveform_times_are_centered_on_zero_with_step_dt() -> None:
    dt = 1e-3
    waveform = crop_waveform(ka=60.0, kc=45.0, J_hz=100.0, dt=dt, truncation_window=0.02)

    npt.assert_allclose(np.diff(waveform["times"]), dt, rtol=1e-8)
    npt.assert_allclose(np.mean(waveform["times"]), 0.0, atol=1e-12)
    npt.assert_allclose(waveform["times"][0], -waveform["times"][-1], atol=1e-12)


def test_crop_eta_rejects_negative_ka() -> None:
    with pytest.raises(ValueError, match="ka must be non-negative"):
        crop_eta(ka=-1.0, kc=0.0, J_hz=100.0)


def test_crop_eta_rejects_non_finite_kc() -> None:
    with pytest.raises(ValueError, match="kc must be finite"):
        crop_eta(ka=60.0, kc=float("nan"), J_hz=100.0)


def test_crop_zeta_rejects_non_positive_J_hz() -> None:
    with pytest.raises(ValueError, match="J_hz must be positive"):
        crop_zeta(ka=60.0, kc=45.0, J_hz=0.0)


def test_crop_zeta_rejects_ka_below_kc() -> None:
    with pytest.raises(ValueError, match="ka must be greater than or equal to kc"):
        crop_zeta(ka=45.0, kc=60.0, J_hz=100.0)


def test_crop_eta_prime_rejects_negative_kc_prime() -> None:
    with pytest.raises(ValueError, match="kc must be non-negative"):
        crop_eta_prime(ka_prime=75.0, kc_prime=-1.0, J_hz=100.0)


def test_crop_limit_Iz_to_2IzSz_rejects_negative_eta() -> None:
    with pytest.raises(ValueError, match="eta must be non-negative"):
        crop_limit_Iz_to_2IzSz(-0.1)


def test_crop_limit_2IzSz_to_Sz_rejects_non_finite_eta_prime() -> None:
    with pytest.raises(ValueError, match="eta_prime must be finite"):
        crop_limit_2IzSz_to_Sz(float("inf"))


def test_crop_limit_Iz_to_Sz_rejects_negative_eta() -> None:
    with pytest.raises(ValueError, match="eta must be non-negative"):
        crop_limit_Iz_to_Sz(-0.5, 0.5)


def test_crop_limit_Iz_to_Sz_rejects_negative_eta_prime() -> None:
    with pytest.raises(ValueError, match="eta_prime must be non-negative"):
        crop_limit_Iz_to_Sz(0.5, -0.5)


def test_crop_limit_single_transition_rejects_negative_eta() -> None:
    with pytest.raises(ValueError, match="eta must be non-negative"):
        crop_limit_single_transition(-0.5, 0.5)


def test_crop_limit_single_transition_rejects_non_finite_eta_prime() -> None:
    with pytest.raises(ValueError, match="eta_prime must be finite"):
        crop_limit_single_transition(0.5, float("nan"))


def test_crop_lossless_limit_rejects_non_positive_J_hz() -> None:
    with pytest.raises(ValueError, match="J_hz must be positive"):
        crop_lossless_limit(ka=60.0, J_hz=-100.0)


def test_single_transition_decomposition_rejects_non_two_spin_system() -> None:
    system = _two_spin(ka=60.0, kc=45.0)
    system.spins.append(Spin(isotope="15N", gamma=1.0, channel="N"))

    with pytest.raises(
        ValueError, match="single_transition_decomposition requires a two-spin system"
    ):
        single_transition_decomposition(system)


def test_single_transition_decomposition_rejects_ka_below_kc() -> None:
    with pytest.raises(ValueError, match="ka must be greater than or equal to kc"):
        single_transition_decomposition(_two_spin(ka=45.0, kc=60.0))


def test_crop_pulse_params_rejects_negative_ka() -> None:
    with pytest.raises(ValueError, match="ka must be non-negative"):
        crop_pulse_params(ka=-1.0, kc=0.0, J_hz=100.0)


def test_crop_waveform_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        crop_waveform(ka=60.0, kc=45.0, J_hz=100.0, dt=0.0, truncation_window=0.01)


def test_crop_waveform_rejects_non_finite_truncation_window() -> None:
    with pytest.raises(ValueError, match="truncation_window must be finite"):
        crop_waveform(ka=60.0, kc=45.0, J_hz=100.0, dt=1e-3, truncation_window=float("inf"))
