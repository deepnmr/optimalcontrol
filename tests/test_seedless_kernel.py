import re
from collections.abc import Callable

import numpy as np
import pytest

from optimalcontrol import _seedless_kernel as kernel
from optimalcontrol._accelerator import RUST_ACCELERATOR_AVAILABLE
from optimalcontrol._types import Array, RealArray


@pytest.mark.parametrize("disable_rust", ["0", "1"])
def test_tiny_field_long_duration_pi_rotation(
    monkeypatch: pytest.MonkeyPatch, disable_rust: str
) -> None:
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)
    iz = np.array([0.0, 0.0, 1.0])

    fidelity, _ = kernel.s2s_value_grad(
        np.array([[1.0, 0.0]]), np.array([0.0]), np.array([1.0]), 1e-200, 5e199, iz, iz
    )

    np.testing.assert_allclose(fidelity, -1.0, rtol=0.0, atol=1e-12)


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


def _error_text(monkeypatch: pytest.MonkeyPatch, backend: str, call: Callable[[], object]) -> str:
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", "1" if backend == "numpy" else "0")
    with pytest.raises(ValueError) as excinfo:
        call()
    return str(excinfo.value)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_s2s_rejects_non_finite_init_bloch(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    if backend == "rust" and not RUST_ACCELERATOR_AVAILABLE:
        pytest.skip("Rust extension is not installed")
    offsets = np.array([0.0, 100.0])
    scales = np.array([1.0])
    targ = np.array([0.0, 0.0, 1.0])
    message = _error_text(
        monkeypatch,
        backend,
        lambda: kernel.s2s_value_grad(
            np.ones((3, 2)), offsets, scales, 1000.0, 1e-5, np.array([np.nan, 0.0, 1.0]), targ
        ),
    )
    assert "finite" in message


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("kernel_name", ["s2s", "suppress"])
def test_kernels_reject_1d_waveform_with_shape_message(
    monkeypatch: pytest.MonkeyPatch, backend: str, kernel_name: str
) -> None:
    if backend == "rust" and not RUST_ACCELERATOR_AVAILABLE:
        pytest.skip("Rust extension is not installed")
    offsets = np.array([0.0, 100.0])
    scales = np.array([1.0])
    iz = np.array([0.0, 0.0, 1.0])
    calls: dict[str, Callable[[], object]] = {
        "s2s": lambda: kernel.s2s_value_grad(np.ones(6), offsets, scales, 1000.0, 1e-5, iz, iz),
        "suppress": lambda: kernel.suppress_perstep_value_grad(
            np.ones(6), offsets, scales, 1000.0, 1e-5
        ),
    }
    message = _error_text(monkeypatch, backend, calls[kernel_name])
    assert re.search(r"shape \(n_steps, 2\)", message)


@pytest.mark.skipif(not RUST_ACCELERATOR_AVAILABLE, reason="Rust extension is not installed")
@pytest.mark.parametrize("case", ["s2s_nan_init", "s2s_1d", "suppress_1d"])
def test_backends_raise_identical_message(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    offsets = np.array([0.0, 100.0])
    scales = np.array([1.0])
    iz = np.array([0.0, 0.0, 1.0])
    calls: dict[str, Callable[[], object]] = {
        "s2s_nan_init": lambda: kernel.s2s_value_grad(
            np.ones((3, 2)), offsets, scales, 1000.0, 1e-5, np.array([np.nan, 0.0, 1.0]), iz
        ),
        "s2s_1d": lambda: kernel.s2s_value_grad(np.ones(6), offsets, scales, 1000.0, 1e-5, iz, iz),
        "suppress_1d": lambda: kernel.suppress_perstep_value_grad(
            np.ones(6), offsets, scales, 1000.0, 1e-5
        ),
    }
    numpy_message = _error_text(monkeypatch, "numpy", calls[case])
    rust_message = _error_text(monkeypatch, "rust", calls[case])
    assert numpy_message == rust_message
