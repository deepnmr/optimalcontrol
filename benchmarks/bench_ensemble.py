"""Microbenchmarks for ensemble GRAPE fidelity scaling.

Run from the repository root with:

    python benchmarks/bench_ensemble.py
"""

import time

import numpy as np
import numpy.typing as npt

from optimalcontrol.ensemble import ensemble_fidelity
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iz
from optimalcontrol.states import normalise_2norm

ComplexArray = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


def _control_problem(member_count: int) -> ControlProblem:
    """Return a one-spin control problem with an RF-power ensemble axis."""
    rho_init = np.array([1.0, 0.0], dtype=np.complex128)
    rho_targ = normalise_2norm(
        np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128)
    )
    if member_count == 1:
        power_levels = [1.0]
    else:
        power_levels = np.linspace(0.85, 1.15, member_count).astype(np.float64).tolist()

    return ControlProblem(
        drifts=[np.complex128(-1j) * 0.2 * Iz()],
        operators=[np.complex128(-1j) * Ix()],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=0.05,
        pwr_levels=power_levels,
        freeze=None,
        fidelity_mode="abs2",
        basis="hilbert",
    )


def _waveform() -> RealArray:
    """Return a deterministic one-channel waveform."""
    return np.array([[0.12], [-0.04], [0.08], [0.03]], dtype=np.float64)


def _time_ensemble(member_count: int) -> float:
    """Return average seconds per ensemble_fidelity call."""
    cp = _control_problem(member_count)
    waveform = _waveform()
    repeats = 300 if member_count <= 4 else 80 if member_count <= 16 else 20

    _ = ensemble_fidelity(cp, waveform)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = ensemble_fidelity(cp, waveform)
    elapsed = time.perf_counter() - start
    return elapsed / float(repeats)


def main() -> None:
    """Print timing results for selected ensemble sizes."""
    print("members,ensemble_fidelity_s")
    for member_count in (1, 4, 16, 64):
        seconds = _time_ensemble(member_count)
        print(f"{member_count},{seconds:.8e}")


if __name__ == "__main__":
    main()
