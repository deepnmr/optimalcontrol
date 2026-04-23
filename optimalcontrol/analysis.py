"""Trajectory analysis helpers for GRAPE optimised pulses."""

import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, forward_propagators, forward_states

Array = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]


def state_trajectory(cp: ControlProblem, wfm: RealArray) -> list[Array]:
    """Return density matrices at each time slice for the first source state.

    Returns a list of length n_steps + 1: [rho_0, rho_1, ..., rho_N].
    For multi-source problems the first source state rho_init[0] is used.
    """
    propagators = forward_propagators(cp, wfm)
    return forward_states(cp.rho_init[0], propagators)


def expectation_values(
    trajectory: list[Array],
    ops: dict[str, Array],
) -> dict[str, RealArray]:
    """Return expectation value of each operator at each time step.

    For a density matrix rho, the expectation value of operator O is
    Re(Tr(O @ rho)).  The returned dict maps each operator name to a
    1-D real array of length len(trajectory).

    Parameters
    ----------
    trajectory:
        List of density matrices or state vectors as returned by
        state_trajectory().
    ops:
        Mapping from label to operator matrix.
    """
    if not trajectory:
        raise ValueError("trajectory must be non-empty")

    result: dict[str, RealArray] = {}
    for name, op in ops.items():
        op_arr = np.asarray(op, dtype=np.complex128)
        values: list[float] = []
        for state in trajectory:
            state_arr = np.asarray(state, dtype=np.complex128)
            if state_arr.ndim == 1:
                ev = float(np.real(np.vdot(state_arr, op_arr @ state_arr)))
            else:
                ev = float(np.real(np.trace(op_arr @ state_arr)))
            values.append(ev)
        result[name] = np.array(values, dtype=np.float64)
    return result


def coherence_order_populations(trajectory: list[Array]) -> RealArray:
    """Return coherence-order populations at each time step.

    Not implemented for the current basis; raises NotImplementedError.
    A future implementation would require explicit product-operator basis
    metadata to decompose density matrices into coherence orders p = -N..N.
    """
    raise NotImplementedError(
        "coherence_order_populations requires explicit product-operator basis "
        "metadata, which is not currently stored in the trajectory."
    )


def correlation_order_populations(trajectory: list[Array]) -> RealArray:
    """Return correlation-order populations at each time step.

    Not implemented for the current basis; raises NotImplementedError.
    A future implementation would require explicit product-operator basis
    metadata to decompose density matrices into spin-correlation orders.
    """
    raise NotImplementedError(
        "correlation_order_populations requires explicit product-operator basis "
        "metadata, which is not currently stored in the trajectory."
    )
