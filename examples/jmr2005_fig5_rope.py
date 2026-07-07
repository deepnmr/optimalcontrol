"""Reproduce Fig. 5 from Khaneja et al., JMR 172 (2005) 296-305.

The paper's Fig. 5 shows a GRAPE-optimized ROPE pulse for
``Iz -> 2IzSz`` in the spin-diffusion limit. The numerical GRAPE solution is
not unique, so this example uses the deterministic finite-time ROPE waveform
already implemented in :mod:`optimalcontrol.rope`, with short boundary hard
pulses sampled over two slices to mimic the finite-amplitude GRAPE pulse.

Saves figure to examples/output/jmr2005_fig5_rope.png.
"""

import matplotlib

matplotlib.use("Agg")

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optimalcontrol.grape import ControlProblem, forward_propagators, forward_states
from optimalcontrol.io import export_bruker_shape as write_bruker_shape
from optimalcontrol.operators import liouvillian_comm, vec
from optimalcontrol.optimizers import lbfgs_grape
from optimalcontrol.rope import rope_finite_efficiency, rope_waveform
from optimalcontrol.spin_system import (
    control_operators,
    drift_hamiltonian,
    relaxation_liouvillian,
    shift_hamiltonian,
    two_spin_system,
)
from optimalcontrol.states import normalise_hs, state_from_label

J_HZ = 194.0
N_RELAX = 1.0
N_STEPS = 75
T_TOTAL = 0.408 / J_HZ
DT = T_TOTAL / N_STEPS
BRUKER_SHAPE_NAME = "jmr2005_fig5_rope.shape"


def _base_rope_waveform() -> npt.NDArray[np.float64]:
    """Return Cartesian ``nu_x/J`` and ``nu_y/J`` finite-time ROPE controls."""
    waveform = rope_waveform(T_TOTAL, N_RELAX, J_HZ, DT)
    nu_over_j = waveform["amplitude"] / (2.0 * math.pi * J_HZ)
    return np.asarray(
        np.column_stack(
            (
                nu_over_j * np.cos(waveform["phase"]),
                nu_over_j * np.sin(waveform["phase"]),
            )
        ),
        dtype=np.float64,
    )


def _add_boundary_pulses(wfm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Sample the ROPE boundary hard pulses over two slices.

    For ``Iz -> 2IzSz``, the 2003 ROPE construction uses initial/final hard
    pulses with flip angle ``asin(u1(0))``. Sampling them over two slices keeps
    the peak amplitude near the finite GRAPE pulse shown in the 2005 figure.
    """
    waveform = np.asarray(wfm, dtype=np.float64).copy()
    u1_initial = rope_waveform(T_TOTAL, N_RELAX, J_HZ, DT)["u1"][0]
    flip_angle = math.asin(float(u1_initial))

    n_boundary = 2
    weights = np.sin(np.linspace(0.0, math.pi, n_boundary + 2)[1:-1])
    weights = weights / float(np.sum(weights))
    amplitude = flip_angle / (2.0 * math.pi * DT * J_HZ)

    waveform[:n_boundary, 1] += amplitude * weights
    waveform[-n_boundary:, 0] += amplitude * weights[::-1]
    return waveform


def _control_problem() -> ControlProblem:
    """Build the JMR 2005 Fig. 5 Liouville-space control problem."""
    system = two_spin_system(
        J_hz=J_HZ,
        kDD=math.pi * J_HZ,
        kCSA_I=0.0,
        kCSA_S=0.0,
        ka=0.0,
        kc=0.0,
        ka_prime=0.0,
        kc_prime=0.0,
    )
    drift = (
        liouvillian_comm(drift_hamiltonian(system) + shift_hamiltonian(system))
        + relaxation_liouvillian(system)
    )
    controls = control_operators(system)
    return ControlProblem(
        drifts=[drift],
        operators=[liouvillian_comm(controls["Ix"]), liouvillian_comm(controls["Iy"])],
        rho_init=[vec(normalise_hs(state_from_label("Iz", 2)))],
        rho_targ=[vec(normalise_hs(state_from_label("2IzSz", 2)))],
        pulse_dt=DT,
        pwr_levels=[2.0 * math.pi * J_HZ, 2.0 * math.pi * J_HZ],
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
    )


def _state_trajectory(
    cp: ControlProblem, wfm: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return trajectories for Iz, Ix, 2IySz, and 2IzSz."""
    propagators = forward_propagators(cp, wfm)
    states = forward_states(cp.rho_init[0], propagators)
    basis_labels = ["Iz", "Ix", "2IySz", "2IzSz"]
    basis = [vec(normalise_hs(state_from_label(label, 2))) for label in basis_labels]
    return np.asarray(
        [[float(np.real(np.vdot(op, state))) for op in basis] for state in states],
        dtype=np.float64,
    )


def _maybe_optimise_waveform(
    cp: ControlProblem,
    waveform: npt.NDArray[np.float64],
    *,
    optimize: bool,
    max_iter: int,
) -> npt.NDArray[np.float64]:
    """Optionally run deterministic L-BFGS GRAPE from the ROPE initial guess."""
    if not optimize:
        return waveform
    result = lbfgs_grape(cp, waveform, m=5, tol_x=1e-8, tol_g=1e-8, max_iter=max_iter)
    print(
        f"  GRAPE optimization: fidelity = {result.fidelity_final:.4f}, "
        f"iterations = {result.n_iter}, reason = {result.reason}"
    )
    return np.asarray(result.wfm_final, dtype=np.float64)


def _export_bruker_shape(waveform: npt.NDArray[np.float64], output_dir: str) -> Path:
    """Write a Bruker-style amplitude/phase shape file for the I channel."""
    amplitude_over_j = np.asarray(np.hypot(waveform[:, 0], waveform[:, 1]), dtype=np.float64)
    max_nu_over_j = float(np.max(amplitude_over_j))
    if max_nu_over_j <= 0.0:
        raise ValueError("waveform amplitude must contain at least one non-zero point")

    amplitude_percent = np.asarray(100.0 * amplitude_over_j / max_nu_over_j, dtype=np.float64)
    phase_deg = np.asarray(np.degrees(np.arctan2(waveform[:, 1], waveform[:, 0])), dtype=np.float64)
    phase_deg = np.mod(phase_deg, 360.0)
    phase_deg[amplitude_percent == 0.0] = 0.0

    return write_bruker_shape(
        Path(output_dir) / BRUKER_SHAPE_NAME,
        "jmr2005_fig5_rope",
        amplitude_percent,
        phase_deg,
        totrot=0.0,
        shape_mode=0,
        extra_tags=[
            f"##$OPTIMALCONTROL_NPOINTS= {N_STEPS}",
            f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {T_TOTAL:.12e}",
            f"##$OPTIMALCONTROL_STEP_DURATION_S= {DT:.12e}",
            f"##$OPTIMALCONTROL_J_HZ= {J_HZ:.12e}",
            f"##$OPTIMALCONTROL_RF_MAX_NU_OVER_J= {max_nu_over_j:.12e}",
            f"##$OPTIMALCONTROL_RF_MAX_NU_HZ= {max_nu_over_j * J_HZ:.12e}",
            "##$OPTIMALCONTROL_NOTE= "
            "Set the Bruker pulse length to TOTAL_DURATION_S; amplitude 100 is RF_MAX_NU_HZ.",
        ],
    )


def run(optimize: bool = False, max_iter: int = 30) -> npt.NDArray[np.float64]:
    """Generate the Fig. 5 reproduction and return numerical snapshot data."""
    waveform = _add_boundary_pulses(_base_rope_waveform())
    cp = _control_problem()
    waveform = _maybe_optimise_waveform(cp, waveform, optimize=optimize, max_iter=max_iter)
    trajectory = _state_trajectory(cp, waveform)

    final_efficiency = float(trajectory[-1, 3])
    theoretical_efficiency = rope_finite_efficiency(T_TOTAL, N_RELAX, J_HZ)
    print(
        "JMR 2005 Fig. 5: "
        f"J = {J_HZ:.1f} Hz, k/J = {N_RELAX:.1f}, "
        f"T = {T_TOTAL * 1e3:.3f} ms = {T_TOTAL * J_HZ:.3f}/J"
    )
    print(
        f"  Simulated <2IzSz> = {final_efficiency:.4f}; "
        f"finite-time ROPE limit = {theoretical_efficiency:.4f}"
    )

    pulse_times = np.arange(N_STEPS, dtype=np.float64) * DT * J_HZ
    trajectory_times = np.arange(N_STEPS + 1, dtype=np.float64) * DT * J_HZ

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)

    ax_pulse = axes[0]
    ax_pulse.plot(pulse_times, waveform[:, 0], color="black", linewidth=1.8)
    ax_pulse.plot(pulse_times, waveform[:, 1], color="black", linewidth=1.8, linestyle="--")
    ax_pulse.text(0.28, 2.1, r"$\nu_x$", fontsize=11)
    ax_pulse.text(0.14, 2.1, r"$\nu_y$", fontsize=11)
    ax_pulse.text(0.97, 0.92, "A", transform=ax_pulse.transAxes, ha="right", va="top", fontsize=16)
    ax_pulse.set_ylabel(r"$\nu_{x,y}/J$")
    ax_pulse.set_ylim(-6.3, 6.3)

    ax_traj = axes[1]
    labels_and_styles = [
        (r"$\langle I_z\rangle$", "-."),
        (r"$\langle I_x\rangle$", "-"),
        (r"$\langle 2I_yS_z\rangle$", "--"),
        (r"$\langle 2I_zS_z\rangle$", ":"),
    ]
    for column, (_label, linestyle) in enumerate(labels_and_styles):
        ax_traj.plot(
            trajectory_times,
            trajectory[:, column],
            color="black",
            linewidth=1.8,
            linestyle=linestyle,
        )
    ax_traj.text(0.16, 0.66, r"$\langle I_z\rangle$", fontsize=10)
    ax_traj.text(0.16, 0.45, r"$\langle I_x\rangle$", fontsize=10)
    ax_traj.text(0.18, 0.19, r"$\langle 2I_yS_z\rangle$", fontsize=10)
    ax_traj.text(0.20, 0.03, r"$\langle 2I_zS_z\rangle$", fontsize=10)
    ax_traj.text(0.97, 0.92, "B", transform=ax_traj.transAxes, ha="right", va="top", fontsize=16)
    ax_traj.set_xlabel(r"$T/J^{-1}$")
    ax_traj.set_ylabel(r"$\eta$")
    ax_traj.set_ylim(-0.08, 1.02)

    for ax in axes:
        ax.set_xlim(0.0, 0.408)
        ax.grid(False)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    shape_path = _export_bruker_shape(waveform, output_dir)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "jmr2005_fig5_rope.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    waveform_with_time = np.column_stack((pulse_times, waveform))
    trajectory_with_time = np.column_stack((trajectory_times, trajectory))
    print(f"Saved Bruker shape {shape_path}")
    return np.concatenate((waveform_with_time.reshape(-1), trajectory_with_time.reshape(-1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="run L-BFGS GRAPE from the deterministic ROPE initial guess before plotting",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="maximum GRAPE iterations used with --optimize",
    )
    args = parser.parse_args()
    run(optimize=args.optimize, max_iter=args.max_iter)
    print("Saved examples/output/jmr2005_fig5_rope.png")
