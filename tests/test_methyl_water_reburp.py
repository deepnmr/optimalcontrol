"""Constraint tests for the smooth REBURP-style methyl/water pulse."""

from pathlib import Path

import numpy as np

from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    RF_MAX_HZ,
    SPECTROMETER_1H_MHZ,
    WATER_PPM,
    evaluate_pulse,
)
from examples.methyl_water_binary_symmetric_180 import (
    signed_amplitude as jagged_signed_amplitude,
)
from examples.methyl_water_reburp_180 import (
    DURATION_S,
    N_STEPS,
    amplitude_phase,
    export_bruker_shape,
    signed_amplitude,
)
from optimalcontrol.bloch import propagate_bloch_ensemble


def test_cached_pulse_enforces_hardware_constraints() -> None:
    signed = signed_amplitude()
    amplitude, phase = amplitude_phase(signed)

    assert signed.shape == (N_STEPS,)
    np.testing.assert_allclose(signed, signed[::-1], atol=0.0)
    assert float(np.max(amplitude) * RF_MAX_HZ) <= RF_MAX_HZ
    assert set(np.unique(phase)).issubset({0.0, 180.0})


def test_cached_pulse_passes_dense_methyl_water_and_artifact_limits() -> None:
    metrics, _ = evaluate_pulse(signed_amplitude(), duration_s=DURATION_S)

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
    # Transverse leakage is the signature of residual 1H chemical-shift
    # evolution across the pulse; a universal rotation refocuses it.
    assert float(np.max(np.abs(fx[:, 1]))) < 5e-3


def test_reburp_waveform_is_smoother_than_the_jagged_sibling() -> None:
    reburp_roughness = float(np.sum(np.diff(signed_amplitude(), 2) ** 2))
    jagged_roughness = float(np.sum(np.diff(jagged_signed_amplitude(), 2) ** 2))

    assert reburp_roughness < jagged_roughness


def test_exported_shape_preserves_binary_phase_and_symmetry(tmp_path: Path) -> None:
    shape_path = export_bruker_shape(signed_amplitude(), tmp_path)
    data_lines = [
        line
        for line in shape_path.read_text(encoding="ascii").splitlines()
        if line and not line.startswith("##")
    ]
    data = np.array([[float(value) for value in line.split(",")] for line in data_lines])

    assert data.shape == (N_STEPS, 2)
    np.testing.assert_allclose(data, data[::-1], atol=0.0)
    assert float(np.max(data[:, 0])) <= 100.0
    assert set(np.unique(data[:, 1])).issubset({0.0, 180.0})
