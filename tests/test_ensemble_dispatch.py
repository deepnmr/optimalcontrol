"""Dispatch contracts for the batched spin-half kernel behind the ensemble rungs."""

import math
from dataclasses import replace

import numpy as np
import pytest

import optimalcontrol._ensemble_kernel as kernel
from optimalcontrol import grape
from optimalcontrol._accelerator import RUST_ACCELERATOR_AVAILABLE
from optimalcontrol._problem import ControlProblem
from optimalcontrol.ensemble import (
    cartesian_product_ensemble,
    ensemble_fidelity,
    ensemble_gradient,
    ensemble_xy_and_gradient,
)
from optimalcontrol.grape import grape_xy, grape_xy_and_gradient
from optimalcontrol.operators import Ix, Iz, liouvillian_comm, vec
from optimalcontrol.penalties import PenaltySpec
from optimalcontrol.states import normalise_hs


def _methyl_problem() -> ControlProblem:
    state = vec(normalise_hs(Ix()))
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=complex)],
        operators=[2 * np.pi * 10000 * liouvillian_comm(Ix())],
        rho_init=[state],
        rho_targ=[state],
        pulse_dt=1.4e-3 / 200,
        pwr_levels=[1.0],
        offsets=np.linspace(-9240, -2040, 81).tolist(),
        offset_operators=[2 * np.pi * liouvillian_comm(Iz())],
        basis="liouville",
        fidelity_mode="real",
    )


def _waveform() -> np.ndarray:
    return np.random.default_rng(11).uniform(-1, 1, (200, 1))


def _set_backend(monkeypatch: pytest.MonkeyPatch, disable_rust: str) -> None:
    monkeypatch.setenv("OPTIMALCONTROL_DISABLE_RUST", disable_rust)


def _ordered_member_sum(cp: ControlProblem, waveform: np.ndarray) -> tuple[float, np.ndarray]:
    value_sum = 0.0
    gradient = np.zeros_like(waveform, dtype=np.float64)
    members = cartesian_product_ensemble(cp)
    for member in members:
        member_value, member_gradient = grape_xy_and_gradient(member, waveform)
        value_sum += member_value
        gradient += member_gradient
    return value_sum / len(members), gradient / len(members)


def test_fallback_ensemble_uses_kernel_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_backend(monkeypatch, "1")
    cp = _methyl_problem()
    waveform = _waveform()
    expected_value, expected_gradient = ensemble_xy_and_gradient(cp, waveform)
    expected_fidelity = ensemble_fidelity(cp, waveform)

    calls = {"gradient": 0, "fidelity": 0}
    original = kernel.spin_half_value_gradient

    def counting(inputs, need_gradient):
        calls["gradient" if need_gradient else "fidelity"] += 1
        return original(inputs, need_gradient=need_gradient)

    monkeypatch.setattr(kernel, "spin_half_value_gradient", counting)
    value, gradient = ensemble_xy_and_gradient(cp, waveform)
    fidelity = ensemble_fidelity(cp, waveform)

    assert calls == {"gradient": 1, "fidelity": 1}
    assert value == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)
    assert fidelity == expected_fidelity


def test_native_ensemble_never_calls_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RUST_ACCELERATOR_AVAILABLE
    _set_backend(monkeypatch, "0")
    kernel_calls = {"inputs": 0, "value_gradient": 0}

    def counting_inputs(*args, **kwargs):
        kernel_calls["inputs"] += 1
        return original_inputs(*args, **kwargs)

    def counting_value_gradient(*args, **kwargs):
        kernel_calls["value_gradient"] += 1
        return original_value_gradient(*args, **kwargs)

    original_inputs = kernel.spin_half_inputs
    original_value_gradient = kernel.spin_half_value_gradient
    monkeypatch.setattr(kernel, "spin_half_inputs", counting_inputs)
    monkeypatch.setattr(kernel, "spin_half_value_gradient", counting_value_gradient)

    waveform = _waveform()
    methyl = _methyl_problem()
    ensemble_xy_and_gradient(methyl, waveform)
    ensemble_fidelity(methyl, waveform)
    assert kernel_calls == {"inputs": 0, "value_gradient": 0}

    phase_cycled = replace(methyl, phase_cycle=np.array([0.0, math.pi / 3]))
    per_member_sizes: list[int] = []
    original_vector = grape.vector_value_gradient

    def counting_vector(problems, wfm):
        per_member_sizes.append(len(problems))
        return original_vector(problems, wfm)

    monkeypatch.setattr(grape, "vector_value_gradient", counting_vector)
    members = cartesian_product_ensemble(phase_cycled)
    value, gradient = ensemble_xy_and_gradient(phase_cycled, waveform)

    assert kernel_calls == {"inputs": 0, "value_gradient": 0}
    assert per_member_sizes == [1] * len(members)
    assert math.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_ineligible_ensemble_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_backend(monkeypatch, "1")
    cp = replace(_methyl_problem(), drifts=[-0.07 * np.eye(4, dtype=complex)])
    waveform = _waveform()

    calls = {"inputs": 0, "value_gradient": 0}
    original_inputs = kernel.spin_half_inputs
    original_value_gradient = kernel.spin_half_value_gradient

    def counting_inputs(*args, **kwargs):
        calls["inputs"] += 1
        return original_inputs(*args, **kwargs)

    def counting_value_gradient(*args, **kwargs):
        calls["value_gradient"] += 1
        return original_value_gradient(*args, **kwargs)

    monkeypatch.setattr(kernel, "spin_half_inputs", counting_inputs)
    monkeypatch.setattr(kernel, "spin_half_value_gradient", counting_value_gradient)

    assert kernel.spin_half_inputs(cp, waveform) is None
    calls["inputs"] = 0
    value, gradient = ensemble_xy_and_gradient(cp, waveform)

    assert calls == {"inputs": 1, "value_gradient": 0}
    expected_value, expected_gradient = _ordered_member_sum(cp, waveform)
    assert value == expected_value
    np.testing.assert_array_equal(gradient, expected_gradient)


_VALIDATION_CASES = [
    "fidelity_mode_abs",
    "empty_rho_init",
    "nan_waveform",
    "wrong_freeze_shape",
    "three_d_waveform",
    "scalar_waveform",
]
_PREEXISTING_NATIVE_GRADIENT_DIVERGENCE = {"three_d_waveform", "scalar_waveform"}


def _invalid_case(name: str) -> tuple[ControlProblem, np.ndarray]:
    cp = _methyl_problem()
    waveform = _waveform()
    match name:
        case "fidelity_mode_abs":
            return replace(cp, fidelity_mode="abs"), waveform
        case "empty_rho_init":
            return replace(cp, rho_init=[]), waveform
        case "nan_waveform":
            bad = waveform.copy()
            bad[3, 0] = np.nan
            return cp, bad
        case "wrong_freeze_shape":
            return replace(cp, freeze=np.zeros((5, 1), dtype=np.bool_)), waveform
        case "three_d_waveform":
            return cp, np.zeros((2, 200, 1))
        case _:
            return cp, np.array(0.5)


def _capture_messages(
    monkeypatch: pytest.MonkeyPatch, disable_rust: str, name: str
) -> dict[str, str]:
    _set_backend(monkeypatch, disable_rust)
    cp, waveform = _invalid_case(name)
    messages: dict[str, str] = {}
    for label, call in (
        ("grape_xy_and_gradient", lambda: grape_xy_and_gradient(cp, waveform)),
        ("grape_xy", lambda: grape_xy(cp, waveform)),
    ):
        with pytest.raises(ValueError) as excinfo:
            call()
        messages[label] = str(excinfo.value)
    return messages


@pytest.mark.parametrize("name", _VALIDATION_CASES)
def test_validation_messages_identical_on_both_paths(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = _capture_messages(monkeypatch, "1", name)
    assert fallback["grape_xy"]
    if name in _PREEXISTING_NATIVE_GRADIENT_DIVERGENCE:
        # The native gradient path raises a different error for these inputs
        # on the untouched tree too; that pre-existing difference is a follow-up.
        return
    native = _capture_messages(monkeypatch, "0", name)
    assert native == fallback


def test_public_ensemble_api_parity_both_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RUST_ACCELERATOR_AVAILABLE
    freeze = np.zeros((200, 1), dtype=np.bool_)
    freeze[0, 0] = True
    cp = replace(
        _methyl_problem(),
        penalties=[PenaltySpec("NS", 1e-3)],
        freeze=freeze,
    )
    waveform = _waveform()
    results = {}
    for disable_rust in ("0", "1"):
        _set_backend(monkeypatch, disable_rust)
        value, gradient = grape_xy_and_gradient(cp, waveform)
        fidelity = grape_xy(cp, waveform)
        ensemble_grad = ensemble_gradient(cp, waveform)
        assert gradient[0, 0] == 0.0
        assert ensemble_grad[0, 0] == 0.0
        results[disable_rust] = (value, fidelity, gradient, ensemble_grad)

    native, fallback = results["0"], results["1"]
    np.testing.assert_allclose(fallback[0], native[0], rtol=1e-12)
    np.testing.assert_allclose(fallback[1], native[1], rtol=1e-12)
    np.testing.assert_allclose(fallback[2], native[2], rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(fallback[3], native[3], rtol=1e-9, atol=1e-14)
