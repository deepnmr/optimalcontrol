"""Constraint tests for the minimum-length REBURP-style methyl/water pulse."""

import numpy as np

from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    SPECTROMETER_1H_MHZ,
    WATER_PPM,
    evaluate_pulse,
)
from examples.methyl_water_reburp_180 import DURATION_US as SMOOTH_SIBLING_US
from examples.methyl_water_reburp_minlength_180 import (
    DURATION_S,
    DURATION_US,
    FRONTIER,
    N_STEPS,
    OPTIMIZED_HALF_AMPLITUDE,
    RF_MAX_HZ,
    amplitude_phase,
    refine_pulse,
    signed_amplitude,
)
from examples.methyl_water_reburp_minpower_180 import DURATION_US as MINPOWER_US
from optimalcontrol.bloch import propagate_bloch_ensemble


def test_cached_pulse_is_the_shortest_smooth_sibling() -> None:
    assert RF_MAX_HZ == 10000.0
    assert DURATION_US == 1800.0
    # Minimum-length end: shorter than both smooth siblings.
    assert DURATION_US < SMOOTH_SIBLING_US
    assert DURATION_US < MINPOWER_US


def test_cached_pulse_enforces_hardware_constraints() -> None:
    signed = signed_amplitude()
    amplitude, phase = amplitude_phase(signed)

    assert signed.shape == (N_STEPS,)
    np.testing.assert_allclose(signed, signed[::-1], atol=0.0)
    assert float(np.max(amplitude)) <= 1.0 + 1e-12
    assert set(np.unique(phase)).issubset({0.0, 180.0})


def test_cached_pulse_passes_dense_limits() -> None:
    metrics, _ = evaluate_pulse(
        signed_amplitude(), duration_s=DURATION_S, rf_max_hz=RF_MAX_HZ
    )

    assert metrics.methyl_x_min >= MIN_METHYL_FIDELITY
    assert metrics.methyl_y_min >= MIN_METHYL_FIDELITY
    assert metrics.methyl_z_min >= MIN_METHYL_FIDELITY
    assert metrics.water_z_min >= MIN_WATER_FIDELITY
    assert metrics.artifact_max_percent <= MAX_ARTIFACT_PERCENT


def test_net_propagator_is_a_universal_180_without_1h_evolution() -> None:
    signed = signed_amplitude()
    waveform = np.column_stack((signed, np.zeros_like(signed)))
    dt = DURATION_S / signed.size
    offsets = (np.linspace(-3.0, 3.0, 301) - WATER_PPM) * SPECTROMETER_1H_MHZ
    scales = np.array([1.0])

    fx = propagate_bloch_ensemble(
        np.array([1.0, 0.0, 0.0]), waveform, offsets, scales, RF_MAX_HZ, dt
    )[0]
    fy = propagate_bloch_ensemble(
        np.array([0.0, 1.0, 0.0]), waveform, offsets, scales, RF_MAX_HZ, dt
    )[0]
    fz = propagate_bloch_ensemble(
        np.array([0.0, 0.0, 1.0]), waveform, offsets, scales, RF_MAX_HZ, dt
    )[0]

    universal = (fx[:, 0] - fy[:, 1] - fz[:, 2]) / 3.0
    assert universal.min() >= MIN_METHYL_FIDELITY
    assert float(np.max(np.abs(fx[:, 1]))) < 5e-3


def test_frontier_minimum_length_matches_the_cached_point() -> None:
    durations = [dur for _, dur in FRONTIER]
    # The cached pulse sits at the shortest duration on the mapped frontier.
    assert min(durations) == DURATION_US
    assert dict(FRONTIER)[RF_MAX_HZ] == DURATION_US


def test_below_minimum_length_fails_the_artifact_threshold() -> None:
    """Re-optimizing below the 1.8 ms floor cannot meet the 0.1% Kay-sideband
    target: a 1.4 ms pulse at the 10 kHz cap (warm-started from the cached
    shape) lands near 0.39%, above the target. This is the manuscript's reason
    for choosing 1.8 ms as the shortest feasible length.
    """
    _, metrics = refine_pulse(1400.0, RF_MAX_HZ, OPTIMIZED_HALF_AMPLITUDE)

    assert not metrics.passes
    assert metrics.artifact_max_percent > MAX_ARTIFACT_PERCENT  # above 0.1%
    # Reproduces ~0.39% (deterministic warm start); bracket loosely.
    assert 0.30 < metrics.artifact_max_percent < 0.50
