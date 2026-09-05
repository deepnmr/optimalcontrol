import numpy as np
import pytest

from optimalcontrol import _seedless_kernel as kernel
from optimalcontrol._accelerator import RUST_ACCELERATOR_AVAILABLE
from optimalcontrol._types import Array, RealArray


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("steps", [1, 2, 17, 64])
def test_suppression_matches_sum_of_independent_prefixes(
    monkeypatch: pytest.MonkeyPatch, backend: str, steps: int
) -> None:
    if backend == "rust" and not RUST_ACCELERATOR_AVAILABLE:
        pytest.skip("Rust extension is not installed")
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1" if backend == "numpy" else "0")
    phases = np.random.default_rng(steps).uniform(-np.pi, np.pi, steps)
    waveform = np.column_stack((np.cos(phases), np.sin(phases)))
    waveform[::3] *= 0.4
    offsets = np.array([-1350.0, 0.0, 2700.0])
    scales = np.array([0.0, 0.8, 1.15])
    iz = np.array([0.0, 0.0, 1.0])

    actual_cost, actual_grad = kernel.suppress_perstep_value_grad(
        waveform, offsets, scales, 2500.0, 2e-5
    )
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    expected_cost = np.zeros(offsets.size * scales.size)
    expected_grad = np.zeros((offsets.size * scales.size, steps))
    for prefix in range(1, steps + 1):
        fidelity, gradient = kernel.s2s_value_grad(
            waveform[:prefix], offsets, scales, 2500.0, 2e-5, iz, iz
        )
        expected_cost += (1.0 - fidelity) / steps
        expected_grad[:, :prefix] -= gradient / steps

    np.testing.assert_allclose(actual_cost, expected_cost, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-12, atol=1e-14)


def test_suppression_uses_one_adjoint_step_per_pulse_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1")
    original = kernel._adjoint_step
    calls = 0

    def counted(v: Array, g: Array, rho: Array, lam: Array) -> tuple[RealArray, Array]:
        nonlocal calls
        calls += 1
        return original(v, g, rho, lam)

    monkeypatch.setattr(kernel, "_adjoint_step", counted)
    steps = 16
    kernel.suppress_perstep_value_grad(
        np.ones((steps, 2)), np.array([0.0]), np.array([1.0]), 2500.0, 2e-5
    )
    assert calls == steps
