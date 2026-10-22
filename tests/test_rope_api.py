"""Direct tests for the analytical ROPE public API in ``optimalcontrol.rope``."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from optimalcontrol.rope import (
    inept_efficiency,
    inept_max_efficiency,
    inept_optimal_time,
    rope_finite_angles,
    rope_g,
    rope_g_inphase,
    rope_hard_pulse_angle,
    rope_j,
    rope_n,
    rope_phase1_control,
    rope_phase3_control,
    rope_switching_time,
    rope_Tcrit,
    rope_trajectory_invariant,
    rope_u_ratio,
    rope_waveform,
)


def test_rope_n_is_kI_over_J() -> None:
    npt.assert_allclose(rope_n(kI=50.0, J=100.0), 0.5, rtol=1e-12)


def test_rope_n_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="kI must be non-negative"):
        rope_n(kI=-1.0, J=100.0)
    with pytest.raises(ValueError, match="J must be positive"):
        rope_n(kI=1.0, J=0.0)
    with pytest.raises(ValueError, match="kI must be finite"):
        rope_n(kI=math.inf, J=100.0)


def test_inept_efficiency_matches_closed_form() -> None:
    J_hz, k_hz, t = 100.0, 20.0, 0.004
    expected = math.exp(-math.pi * k_hz * t) * math.sin(math.pi * J_hz * t)

    npt.assert_allclose(inept_efficiency(t, J_hz, k_hz), expected, rtol=1e-12)
    npt.assert_allclose(inept_efficiency(0.0, J_hz, k_hz), 0.0, atol=1e-15)


def test_inept_efficiency_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="t must be non-negative"):
        inept_efficiency(-1e-3, 100.0, 1.0)
    with pytest.raises(ValueError, match="J_hz must be positive"):
        inept_efficiency(1e-3, 0.0, 1.0)
    with pytest.raises(ValueError, match="k_hz must be non-negative"):
        inept_efficiency(1e-3, 100.0, -1.0)


def test_inept_optimal_time_maximises_efficiency_on_grid() -> None:
    n, J_hz = 1.0, 100.0
    t_star = inept_optimal_time(n, J_hz)
    grid = np.linspace(0.0, 3.0 / J_hz, 30001)
    values = np.array([inept_efficiency(float(t), J_hz, n * J_hz) for t in grid])

    npt.assert_allclose(t_star, math.atan2(1.0, n) / (math.pi * J_hz), rtol=1e-12)
    assert inept_efficiency(t_star, J_hz, n * J_hz) >= values.max() - 1e-12
    npt.assert_allclose(inept_max_efficiency(n, J_hz), values.max(), atol=1e-6)


def test_inept_optimal_time_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="x must be non-negative"):
        inept_optimal_time(-1.0, 100.0)


def test_rope_g_inphase_is_square_of_rope_g() -> None:
    for n in (0.0, 0.3, 1.0, 2.5):
        npt.assert_allclose(rope_g_inphase(n), rope_g(n) ** 2, rtol=1e-12)


def test_rope_u_ratio_is_g_r1_over_r2() -> None:
    n = 1.0
    npt.assert_allclose(rope_u_ratio(r1=0.3, r2=0.6, n=n), rope_g(n) * 0.5, rtol=1e-12)
    npt.assert_allclose(rope_u_ratio(r1=0.0, r2=0.6, n=n), 0.0, atol=0.0)
    assert rope_u_ratio(r1=0.3, r2=0.0, n=n) == math.inf
    with pytest.raises(ValueError, match="n must be non-negative"):
        rope_u_ratio(r1=0.0, r2=0.6, n=-1.0)


def test_rope_trajectory_invariant_is_ratio_of_expectations() -> None:
    npt.assert_allclose(
        rope_trajectory_invariant(expect_2IySz=0.4, expect_Ix=0.8, n=1.0), 0.5, rtol=1e-12
    )
    assert rope_trajectory_invariant(expect_2IySz=0.0, expect_Ix=0.8, n=1.0) == 0.0
    assert rope_trajectory_invariant(expect_2IySz=0.4, expect_Ix=0.0, n=1.0) == math.inf
    with pytest.raises(ValueError, match="n must be non-negative"):
        rope_trajectory_invariant(expect_2IySz=0.4, expect_Ix=0.8, n=-1.0)


def test_rope_j_limits() -> None:
    npt.assert_allclose(rope_j(0.5, 0.0), 1.0, rtol=1e-12)
    npt.assert_allclose(rope_j(0.0, 0.0), 0.0, atol=0.0)
    npt.assert_allclose(rope_j(1e3, 1.0), rope_g(1.0) ** 2, rtol=1e-8)
    with pytest.raises(ValueError, match="s must be non-negative"):
        rope_j(-1.0, 1.0)


def test_rope_finite_angles_satisfy_switching_identity() -> None:
    n, J_hz = 1.0, 100.0
    T = 0.263 / J_hz
    h1, h2 = rope_finite_angles(T, n, J_hz)
    s = rope_switching_time(T, n, J_hz)

    # Appendix B Eq. (10): (h2 - h1) + 2 * pi * J * s = pi * J * T
    npt.assert_allclose((h2 - h1) + 2.0 * math.pi * J_hz * s, math.pi * J_hz * T, rtol=1e-9)
    assert 0.0 < h1 < h2 < math.pi / 2.0 + 1.0


def test_rope_finite_angles_unrelaxed_limit() -> None:
    # T_crit(n=0) = 1/(2J), so T = 1/J is a valid finite-time duration.
    h1, h2 = rope_finite_angles(T=1.0 / 100.0, n=0.0, J_hz=100.0)
    npt.assert_allclose((h1, h2), (math.pi / 4.0, math.pi / 4.0), rtol=1e-12)


def test_rope_phase1_control_boundary_and_bounds() -> None:
    n, J_hz = 1.0, 100.0
    T = 0.263 / J_hz
    h1, h2 = rope_finite_angles(T, n, J_hz)
    s = math.pi * J_hz * rope_switching_time(T, n, J_hz)

    assert rope_phase1_control(s, s, h1, h2, n) == (1.0, 1.0)
    u1, u2 = rope_phase1_control(0.0, s, h1, h2, n)
    assert 0.0 <= u1 <= 1.0
    assert u2 == 1.0
    with pytest.raises(ValueError, match="t must be within the phase-I interval"):
        rope_phase1_control(s + 1.0, s, h1, h2, n)
    with pytest.raises(ValueError, match="h2 produces a zero tangent"):
        rope_phase1_control(0.0, s, h1, 0.0, n)


def test_rope_phase3_control_mirrors_phase1() -> None:
    n, J_hz = 1.0, 100.0
    T = 0.263 / J_hz
    h1, h2 = rope_finite_angles(T, n, J_hz)
    s = math.pi * J_hz * rope_switching_time(T, n, J_hz)
    scaled_T = math.pi * J_hz * T

    assert rope_phase3_control(0.0, scaled_T, s, h1, h2, n) == (1.0, 1.0)
    u1_end, u2_end = rope_phase3_control(scaled_T, scaled_T, s, h1, h2, n)
    u1_start, _ = rope_phase1_control(0.0, s, h1, h2, n)
    assert u1_end == 1.0
    npt.assert_allclose(u2_end, u1_start, rtol=1e-12)
    with pytest.raises(ValueError, match="t must be within the transfer duration"):
        rope_phase3_control(scaled_T + 1.0, scaled_T, s, h1, h2, n)
    with pytest.raises(ValueError, match="s must not exceed T/2"):
        rope_phase3_control(0.0, scaled_T, scaled_T, h1, h2, n)


def test_rope_hard_pulse_angle_matches_documented_example() -> None:
    # examples/rope_hard_pulse.py: n = 1, J = 100 Hz, T = 0.263 / J.
    n, J_hz = 1.0, 100.0
    T = 0.263 / J_hz
    h1, h2 = rope_finite_angles(T, n, J_hz)
    s = math.pi * J_hz * rope_switching_time(T, n, J_hz)
    u1_at_0, _ = rope_phase1_control(0.0, s, h1, h2, n)

    npt.assert_allclose(rope_hard_pulse_angle(u1_at_0, 1.0), 55.14, atol=0.01)


def test_rope_hard_pulse_angle_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="amplitude must be positive"):
        rope_hard_pulse_angle(0.5, 0.0)
    with pytest.raises(ValueError, match=r"h / amplitude must be within \[0, 1\]"):
        rope_hard_pulse_angle(2.0, 1.0)


def test_rope_waveform_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        rope_waveform(T=0.01, n=1.0, J_hz=100.0, dt=0.0)
    with pytest.raises(ValueError, match="T must be positive"):
        rope_waveform(T=0.0, n=1.0, J_hz=100.0, dt=1e-4)
    with pytest.raises(ValueError, match="J_hz must be positive"):
        rope_Tcrit(n=1.0, J_hz=0.0)
