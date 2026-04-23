"""Tests for analytical ROPE helper formulas."""

import numpy.testing as npt

from optimalcontrol.rope import (
    inept_max_efficiency,
    rope_g,
    rope_gain_over_inept,
    rope_V,
)


def test_rope_g_no_relaxation_limit() -> None:
    npt.assert_allclose(rope_g(n=0.0), 1.0, rtol=1e-8)


def test_inept_max_efficiency_no_relaxation_limit() -> None:
    npt.assert_allclose(inept_max_efficiency(n=0.0, J_hz=100.0), 1.0, rtol=1e-8)


def test_rope_gain_over_inept_no_relaxation_limit() -> None:
    npt.assert_allclose(rope_gain_over_inept(n=0.0, J_hz=100.0), 1.0, rtol=1e-8)


def test_rope_V_unit_second_coordinate() -> None:
    npt.assert_allclose(rope_V(r1=0.0, r2=1.0, n=1.0), 1.0, rtol=1e-8)
