"""Constraint tests for the symmetric binary-phase methyl/water pulse."""

from pathlib import Path

import numpy as np
import pytest

import examples.methyl_water_binary_symmetric_180 as pulse_module
from examples.methyl_water_binary_symmetric_180 import (
    MAX_ARTIFACT_PERCENT,
    MIN_METHYL_FIDELITY,
    MIN_WATER_FIDELITY,
    N_STEPS,
    RF_MAX_HZ,
    amplitude_phase,
    evaluate_pulse,
    export_bruker_shape,
    signed_amplitude,
)


def test_cached_pulse_enforces_hardware_constraints() -> None:
    signed = signed_amplitude()
    amplitude, phase = amplitude_phase(signed)

    assert signed.shape == (N_STEPS,)
    np.testing.assert_allclose(signed, signed[::-1], atol=0.0)
    assert float(np.max(amplitude) * RF_MAX_HZ) <= RF_MAX_HZ
    assert set(np.unique(phase)).issubset({0.0, 180.0})


def test_cached_pulse_passes_dense_methyl_water_and_artifact_limits() -> None:
    metrics, _ = evaluate_pulse(signed_amplitude())

    assert metrics.methyl_x_min >= MIN_METHYL_FIDELITY
    assert metrics.methyl_y_min >= MIN_METHYL_FIDELITY
    assert metrics.methyl_z_min >= MIN_METHYL_FIDELITY
    assert metrics.water_z_min >= MIN_WATER_FIDELITY
    assert metrics.artifact_max_percent <= MAX_ARTIFACT_PERCENT


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


def test_duration_search_preserves_the_cached_boundary_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def evaluate_seed_without_refinement(
        duration_us: float,
        half_seed: np.ndarray,  # type: ignore[type-arg]
        max_outer: int = 8,
        max_iter: int = 160,
    ) -> tuple[np.ndarray, pulse_module.PulseMetrics]:  # type: ignore[type-arg]
        del max_outer, max_iter
        metrics, _ = pulse_module.evaluate_pulse(
            pulse_module.signed_amplitude(half_seed), duration_s=duration_us * 1e-6
        )
        return half_seed.copy(), metrics

    monkeypatch.setattr(pulse_module, "refine_duration", evaluate_seed_without_refinement)
    duration_us, _, audit = pulse_module.search_minimum_duration()

    assert duration_us == 1740.0
    assert next(metrics for duration, metrics in audit if duration == 1740.0).passes
    assert not next(metrics for duration, metrics in audit if duration == 1735.0).passes
    assert not next(metrics for duration, metrics in audit if duration == 1745.0).passes
