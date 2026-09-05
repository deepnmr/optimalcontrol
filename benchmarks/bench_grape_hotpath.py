"""Hot-path benchmark for GRAPE fidelity and gradient evaluation.

Run from the repository root with:

    python benchmarks/bench_grape_hotpath.py
"""

from collections.abc import Callable
from statistics import median
from timeit import repeat

import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, grape_gradient, grape_xy
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm
from optimalcontrol.states import normalise_hs, state_from_label

RealArray = npt.NDArray[np.float64]

N_STEPS = 72
RF_MAX_HZ = 7_500.0
DT = 540.0e-6 / N_STEPS


def _single_problem() -> ControlProblem:
    """Return a single-member Liouville problem matching the examples."""
    two_pi = 2.0 * np.pi
    drift = liouvillian_comm(two_pi * 1_000.0 * Iz())
    op_x = liouvillian_comm(Ix())
    op_y = liouvillian_comm(Iy())
    rho_init = normalise_hs(-state_from_label("Iy", 1)).reshape(-1)
    rho_targ = normalise_hs(state_from_label("Iy", 1)).reshape(-1)
    return ControlProblem(
        drifts=[drift],
        operators=[op_x, op_y],
        rho_init=[rho_init],
        rho_targ=[rho_targ],
        pulse_dt=DT,
        pwr_levels=[two_pi * RF_MAX_HZ, two_pi * RF_MAX_HZ],
        freeze=None,
        fidelity_mode="real",
    )


def _ensemble_problem() -> ControlProblem:
    """Return a 5-offset x 3-power ensemble problem."""
    cp = _single_problem()
    cp.offsets = np.linspace(-10_000.0, 10_000.0, 5).tolist()
    cp.offset_operators = [liouvillian_comm(2.0 * np.pi * Iz())]
    cp.drifts = [np.zeros_like(cp.drifts[0])]
    return cp


def _waveform() -> RealArray:
    """Return a deterministic two-channel waveform."""
    rng = np.random.default_rng(7)
    return np.asarray(0.5 * rng.standard_normal((N_STEPS, 2)), dtype=np.float64)


def _density_problem() -> ControlProblem:
    cp = _single_problem()
    cp.drifts = [-1j * 2.0 * np.pi * 1000.0 * Iz()]
    cp.operators = [-1j * Ix(), -1j * Iy()]
    cp.rho_init = [normalise_hs(-Iy())]
    cp.rho_targ = [normalise_hs(Iy())]
    return cp


def _time_call(fn: Callable[[], float | RealArray], repeats: int) -> float:
    fn()
    return median(repeat(fn, number=repeats, repeat=5)) / repeats


def main() -> None:
    wfm = _waveform()
    single = _single_problem()
    ens = _ensemble_problem()
    density = _density_problem()

    print("case,seconds_per_call")
    print(f"single_fidelity,{_time_call(lambda: grape_xy(single, wfm), 20):.6e}")
    print(f"single_gradient,{_time_call(lambda: grape_gradient(single, wfm), 3):.6e}")
    print(f"ensemble_fidelity,{_time_call(lambda: grape_xy(ens, wfm), 5):.6e}")
    print(f"ensemble_gradient,{_time_call(lambda: grape_gradient(ens, wfm), 1):.6e}")
    print(f"density_fidelity,{_time_call(lambda: grape_xy(density, wfm), 20):.6e}")
    print(f"density_gradient,{_time_call(lambda: grape_gradient(density, wfm), 3):.6e}")


if __name__ == "__main__":
    main()
