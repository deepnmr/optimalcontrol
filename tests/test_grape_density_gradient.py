import numpy as np
import pytest

from optimalcontrol.grape import ControlProblem, grape_xy, grape_xy_and_gradient


@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("mode", ["real", "imag", "abs2"])
@pytest.mark.parametrize("relaxation", [0.0, 0.2])
def test_density_gradient_matches_finite_differences(
    dim: int, mode: str, relaxation: float
) -> None:
    rng = np.random.default_rng(19)
    matrices = rng.normal(size=(4, dim, dim)) + 1j * rng.normal(size=(4, dim, dim))
    generators = (matrices - matrices.conj().swapaxes(-1, -2)) / 2.0
    initial = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    target = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    initial /= np.linalg.norm(initial)
    target /= np.linalg.norm(target)
    cp = ControlProblem(
        drifts=[generators[0] - relaxation * np.eye(dim)],
        operators=list(generators[1:]),
        rho_init=[initial, initial.T],
        rho_targ=[target, target.T],
        pulse_dt=0.08,
        pwr_levels=[0.7, 0.9, 1.1],
        freeze=None,
        fidelity_mode=mode,
        basis="dense",
    )
    waveform = rng.uniform(-0.5, 0.5, size=(5, 3))
    value, gradient = grape_xy_and_gradient(cp, waveform)
    assert value == pytest.approx(grape_xy(cp, waveform), abs=1e-14)
    numeric = np.empty_like(waveform)
    eps = 1e-6
    for index in np.ndindex(waveform.shape):
        plus, minus = waveform.copy(), waveform.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = (grape_xy(cp, plus) - grape_xy(cp, minus)) / (2.0 * eps)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-9)
