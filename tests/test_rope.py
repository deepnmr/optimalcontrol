"""Tests for analytical ROPE helper formulas."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from optimalcontrol.rope import (
    inept_max_efficiency,
    rope_finite_efficiency,
    rope_g,
    rope_gain_over_inept,
    rope_switching_time,
    rope_Tcrit,
    rope_V,
    rope_waveform,
)


def test_rope_g_no_relaxation_limit() -> None:
    npt.assert_allclose(rope_g(n=0.0), 1.0, rtol=1e-8)


def test_inept_max_efficiency_no_relaxation_limit() -> None:
    npt.assert_allclose(inept_max_efficiency(n=0.0, J_hz=100.0), 1.0, rtol=1e-8)


def test_rope_gain_over_inept_no_relaxation_limit() -> None:
    npt.assert_allclose(rope_gain_over_inept(n=0.0, J_hz=100.0), 1.0, rtol=1e-8)


def test_rope_V_unit_second_coordinate() -> None:
    npt.assert_allclose(rope_V(r1=0.0, r2=1.0, n=1.0), 1.0, rtol=1e-8)


def test_rope_Tcrit_matches_appendix_b_value() -> None:
    expected = math.atan2(1.0, 2.0) / (math.pi * 100.0)

    npt.assert_allclose(rope_Tcrit(n=1.0, J_hz=100.0), expected, rtol=1e-12)


def test_rope_finite_efficiency_approaches_unconstrained_limit() -> None:
    n = 1.0
    J_hz = 100.0

    finite_efficiency = rope_finite_efficiency(T=10.0 / J_hz, n=n, J_hz=J_hz)

    npt.assert_allclose(finite_efficiency, rope_g(n), rtol=1e-8)


def test_rope_waveform_samples_phase_ii_controls() -> None:
    n = 1.0
    J_hz = 100.0
    T = 0.263 / J_hz
    dt = T / 20.0

    waveform = rope_waveform(T=T, n=n, J_hz=J_hz, dt=dt)

    expected_length = math.ceil(T / dt)
    for values in waveform.values():
        assert len(values) == expected_length

    switching_time = rope_switching_time(T=T, n=n, J_hz=J_hz)
    times = waveform["times"]
    phase2_mask = (times >= switching_time) & (times <= T - switching_time)

    assert np.any(phase2_mask)
    npt.assert_allclose(waveform["u1"][phase2_mask], 1.0, rtol=1e-12)
    npt.assert_allclose(waveform["u2"][phase2_mask], 1.0, rtol=1e-12)


def test_rope_switching_time_rejects_duration_at_or_below_Tcrit() -> None:
    Tcrit = rope_Tcrit(n=1.0, J_hz=100.0)

    with pytest.raises(ValueError, match="T must be greater than T_crit"):
        rope_switching_time(T=Tcrit, n=1.0, J_hz=100.0)

    with pytest.raises(ValueError, match="T must be greater than T_crit"):
        rope_switching_time(T=Tcrit * 0.5, n=1.0, J_hz=100.0)
