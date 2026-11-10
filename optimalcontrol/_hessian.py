"""Second-order fidelity and state propagation helpers."""

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol._problem import ControlProblem, _raise_invalid_mode
from optimalcontrol._propagation import (
    _apply_derivative_propagator,
    _apply_propagator,
    _apply_second_derivative_propagator,
    _control_direction_stack,
    _dir_diff_generator_expm,
    _second_dir_diff_generator_expm,
    _slice_generator_stack,
)
from optimalcontrol._types import Array, RealArray


def _fidelity_second_directional_derivative(
    rho_final: Array,
    rho_targ: Array,
    d_rho_a: Array,
    d_rho_b: Array,
    dd_rho: Array,
    mode: str,
) -> float:
    """Return the fidelity mixed second derivative for final-state variations."""
    rho_final_arr = np.asarray(rho_final, dtype=np.complex128)
    rho_targ_arr = np.asarray(rho_targ, dtype=np.complex128)
    d_a_arr = np.asarray(d_rho_a, dtype=np.complex128)
    d_b_arr = np.asarray(d_rho_b, dtype=np.complex128)
    dd_arr = np.asarray(dd_rho, dtype=np.complex128)
    for name, array in (
        ("rho_targ", rho_targ_arr),
        ("d_rho_a", d_a_arr),
        ("d_rho_b", d_b_arr),
        ("dd_rho", dd_arr),
    ):
        if array.shape != rho_final_arr.shape:
            raise ValueError(
                f"{name} shape {array.shape} must match rho_final shape {rho_final_arr.shape}"
            )

    dd_overlap = np.complex128(np.vdot(rho_targ_arr, dd_arr))
    if mode == "abs2":
        overlap = np.complex128(np.vdot(rho_targ_arr, rho_final_arr))
        d_overlap_a = np.complex128(np.vdot(rho_targ_arr, d_a_arr))
        d_overlap_b = np.complex128(np.vdot(rho_targ_arr, d_b_arr))
        return float(
            2.0 * np.real(d_overlap_b.conjugate() * d_overlap_a + overlap.conjugate() * dd_overlap)
        )
    return _mode_value(dd_overlap, mode)


def _mode_value(overlap: np.complex128, mode: str) -> float:
    """Return the scalar fidelity for one target-final overlap."""
    if mode == "real":
        return float(np.real(overlap))
    if mode == "imag":
        return float(np.imag(overlap))
    if mode == "abs2":
        return float(np.real(overlap.conjugate() * overlap))
    _raise_invalid_mode("mode")


def _hessian_slice_propagators(
    cp: ControlProblem, effective_waveform: RealArray
) -> tuple[list[Array], list[list[Array]], list[list[list[Array]]]]:
    """Return per-slice propagators with their first and second derivatives."""
    n_channels = effective_waveform.shape[1]
    propagators: list[Array] = []
    derivative_propagators: list[list[Array]] = []
    second_derivative_propagators: list[list[list[Array]]] = []
    control_directions = list(_control_direction_stack(cp))
    generators = _slice_generator_stack(cp, effective_waveform)
    for generator in generators:
        propagator = np.asarray(expm(generator * cp.pulse_dt), dtype=np.complex128)
        propagators.append(propagator)
        derivative_propagators.append(
            [
                _dir_diff_generator_expm(generator, control_direction, cp.pulse_dt)
                for control_direction in control_directions
            ]
        )
        second_derivative_propagators.append(
            [
                [
                    _second_dir_diff_generator_expm(
                        generator,
                        control_directions[channel_a],
                        control_directions[channel_b],
                        cp.pulse_dt,
                    )
                    for channel_b in range(n_channels)
                ]
                for channel_a in range(n_channels)
            ]
        )
    return propagators, derivative_propagators, second_derivative_propagators


def _hessian_state_sweep(
    rho_init: Array,
    propagators: list[Array],
    derivative_propagators: list[list[Array]],
    second_derivative_propagators: list[list[list[Array]]],
    n_channels: int,
    n_params: int,
) -> tuple[Array, list[Array], list[list[Array]]]:
    """Propagate one state with all first and second parameter variations."""
    current = np.asarray(rho_init, dtype=np.complex128).copy()
    zero_state = np.zeros_like(current, dtype=np.complex128)
    first_states = [zero_state.copy() for _ in range(n_params)]
    second_states = [[zero_state.copy() for _ in range(n_params)] for _ in range(n_params)]

    for step_index in range(len(propagators)):
        propagator = propagators[step_index]
        next_current = _apply_propagator(current, propagator)
        next_first_states = [
            _apply_propagator(first_state, propagator) for first_state in first_states
        ]
        next_second_states = [
            [_apply_propagator(second_states[row][col], propagator) for col in range(n_params)]
            for row in range(n_params)
        ]

        for channel_index in range(n_channels):
            param_index = step_index * n_channels + channel_index
            d_prop = derivative_propagators[step_index][channel_index]
            for other_index in range(n_params):
                cross_term = _apply_derivative_propagator(
                    first_states[other_index],
                    propagator,
                    d_prop,
                )
                next_second_states[param_index][other_index] += cross_term
                next_second_states[other_index][param_index] += cross_term
            next_first_states[param_index] += _apply_derivative_propagator(
                current,
                propagator,
                d_prop,
            )

        for channel_a in range(n_channels):
            param_a = step_index * n_channels + channel_a
            for channel_b in range(n_channels):
                param_b = step_index * n_channels + channel_b
                next_second_states[param_a][param_b] += _apply_second_derivative_propagator(
                    current,
                    propagator,
                    derivative_propagators[step_index][channel_a],
                    derivative_propagators[step_index][channel_b],
                    second_derivative_propagators[step_index][channel_a][channel_b],
                )

        current = next_current
        first_states = next_first_states
        second_states = next_second_states

    return current, first_states, second_states


def _zero_frozen_hessian(
    hessian: RealArray,
    freeze: npt.NDArray[np.bool_] | None,
) -> None:
    """Zero Hessian rows and columns for frozen waveform entries, in place."""
    if freeze is None:
        return
    freeze_mask = np.asarray(freeze, dtype=np.bool_).reshape(hessian.shape[0])
    hessian[freeze_mask, :] = 0.0
    hessian[:, freeze_mask] = 0.0
