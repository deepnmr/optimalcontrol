"""Constraint tests for the minimum-power REBURP-style methyl/water pulse."""

import numpy as np

from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    SPECTROMETER_1H_MHZ,
    WATER_PPM,
    evaluate_pulse,
)
from examples.methyl_water_reburp_180 import RF_MAX_HZ as SIBLING_RF_MAX_HZ
from examples.methyl_water_reburp_minpower_180 import (
    DURATION_S,
    FRONTIER,
    N_STEPS,
    RF_MAX_HZ,
    amplitude_phase,
    signed_amplitude,
)
from optimalcontrol.bloch import propagate_bloch_ensemble


def test_cached_pulse_uses_a_lower_peak_field_than_the_10khz_siblings() -> None:
    # The whole point of this pulse: a smaller maximum RF amplitude.
    assert RF_MAX_HZ < SIBLING_RF_MAX_HZ
    assert RF_MAX_HZ == 6000.0


def test_cached_pulse_enforces_hardware_constraints() -> None:
    signed = signed_amplitude()
    amplitude, phase = amplitude_phase(signed)

    assert signed.shape == (N_STEPS,)
    np.testing.assert_allclose(signed, signed[::-1], atol=0.0)
    # Amplitude is a fraction of the (reduced) peak field and never exceeds it.
    assert float(np.max(amplitude)) <= 1.0 + 1e-12
    assert set(np.unique(phase)).issubset({0.0, 180.0})


def test_cached_pulse_passes_dense_limits_at_the_reduced_power() -> None:
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
    # Transverse leakage signals residual 1H chemical-shift evolution; a
    # universal rotation refocuses it even at the reduced peak field.
    assert float(np.max(np.abs(fx[:, 1]))) < 5e-3


def test_frontier_is_monotone_and_brackets_the_cached_point() -> None:
    powers = [rf for rf, _ in FRONTIER]
    durations = [dur for _, dur in FRONTIER]

    # Scanned from high to low peak field.
    assert powers == sorted(powers, reverse=True)
    # Lower peak field never needs a shorter duration (a real trade-off).
    assert durations == sorted(durations)
    # The cached pulse is the minimum-power end of the mapped frontier.
    assert min(powers) == RF_MAX_HZ
    assert dict(FRONTIER)[RF_MAX_HZ] == 2600.0
