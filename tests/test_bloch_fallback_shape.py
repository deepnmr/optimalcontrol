"""The NumPy Bloch fallback performs exactly one rotation call per waveform step."""

from __future__ import annotations

import numpy as np
import pytest

from optimalcontrol import bloch
from optimalcontrol.bloch import propagate_bloch_ensemble


def test_fallback_calls_rotate_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    rng = np.random.default_rng(3)
    waveform = rng.uniform(-1.0, 1.0, (17, 2))
    offsets = np.linspace(-3000.0, 3000.0, 23)
    scales = np.array([0.8, 1.0, 1.2])
    initial = np.array([0.0, 0.0, 1.0])
    expected = propagate_bloch_ensemble(initial, waveform, offsets, scales, 5000.0, 1e-5)

    original = bloch._rotate
    calls = 0

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(bloch, "_rotate", counted)
    actual = propagate_bloch_ensemble(initial, waveform, offsets, scales, 5000.0, 1e-5)
    assert calls == 17
    assert np.array_equal(actual, expected)
