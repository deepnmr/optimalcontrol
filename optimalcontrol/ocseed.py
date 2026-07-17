"""Seedless-style band/restraint front-end for isolated spin-1/2 pulse design.

Implements the "Seedless" formalism of Buchanan et al., *Nature Communications*
16, 7276 (2025), on top of this package's optimal-control engine. A pulse is
described by a set of chemical-shift *bands* (in ppm), each carrying one of four
*restraints*:

* ``"universal"`` -- a full Bloch-sphere rotation ``U`` (3-axis control),
  realised as the three cardinal state-to-state transfers ``X->UX``, ``Y->UY``,
  ``Z->UZ`` (Supplementary Note 2.4).
* ``"s2s"`` -- a single state-to-state transfer, e.g. ``z -> -z`` (Note 2.2/2.3).
* ``"xycite"`` -- take ``Iz`` into the transverse plane without caring where,
  minimising the squared residual ``Iz`` component (Note 2.6).
* ``"suppress"`` -- keep ``Iz`` on ``Iz`` (water hold). Defaults to the
  *end-of-pulse* form; set ``per_step=True`` on the band for the paper's per-step
  ``n^2/2`` suppression (hold ``Iz`` after every prefix, Note 2.7).

The optimisation variable is a single constant-amplitude, phase-only waveform
(the paper's preferred mode). B1 inhomogeneity is handled by a weighted average
over ``b1_scales`` (Note 2.8). By default (``fast=True``) values and exact
gradients come from the analytic scaled-unitary spin-1/2 kernel
(:mod:`optimalcontrol._seedless_kernel`, SI Note 2.9/2.11/2.12); ``fast=False``
routes them through the general 4x4 Liouville GRAPE core
(:func:`grape_xy_and_gradient`) as a cross-check. An independent Bloch forward
model (:func:`propagate_bloch_ensemble`) is used for worst-case verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from optimalcontrol import _seedless_kernel as kernel
from optimalcontrol._seedless_kernel import bloch_operator as _bloch_operator
from optimalcontrol._types import Array as ComplexArray
from optimalcontrol._types import RealArray
from optimalcontrol.bloch import propagate_bloch_ensemble
from optimalcontrol.grape import (
    ControlProblem,
    ampl_phase_to_xy,
    grape_xy_and_gradient,
    phase_only_gradient,
)
from optimalcontrol.io import export_bruker_shape
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.states import normalise_hs

_CARDINAL: dict[str, RealArray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}
VALID_RESTRAINTS = frozenset({"universal", "s2s", "xycite", "suppress"})


def _axis_bloch(label: str) -> RealArray:
    """Return the signed cardinal Bloch vector for a label like ``"-z"``."""
    text = label.strip().lower()
    sign = 1.0
    if text and text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    if text not in _CARDINAL:
        raise ValueError(f"axis label must be one of x, y, z (optionally signed), got {label!r}")
    return np.asarray(sign * _CARDINAL[text], dtype=np.float64)


def _rotation_matrix(axis_label: str, angle_deg: float) -> RealArray:
    """Return the 3x3 right-handed Bloch rotation about a cardinal axis."""
    axis = _axis_bloch(axis_label)
    theta = math.radians(angle_deg)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.asarray(
        np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Band:
    """One chemical-shift band and the restraint acting on the spins within it.

    ``ppm_lo``/``ppm_hi`` bound the band; ``n_offsets`` spins are linearly spaced
    across it. ``restraint`` selects the transform. ``init``/``targ`` give axis
    labels for ``"s2s"`` (e.g. ``"z"`` -> ``"-z"``); ``rotation`` gives the target
    unitary for ``"universal"`` as ``(axis_label, angle_deg)`` (e.g.
    ``("x", 180.0)``). ``weight`` scales the band's contribution to the cost.
    ``per_step`` (``"suppress"`` only) selects the paper's per-step ``n^2/2``
    suppression (hold ``Iz`` after every prefix, Supplementary Note 2.7) instead
    of the cheaper end-of-pulse hold.
    """

    ppm_lo: float
    ppm_hi: float
    restraint: str
    n_offsets: int = 20
    init: str | None = None
    targ: str | None = None
    rotation: tuple[str, float] | None = None
    weight: float = 1.0
    per_step: bool = False

    def __post_init__(self) -> None:
        if self.restraint not in VALID_RESTRAINTS:
            raise ValueError(f"restraint must be one of {sorted(VALID_RESTRAINTS)}")
        if self.ppm_hi < self.ppm_lo:
            raise ValueError("ppm_hi must be >= ppm_lo")
        if self.n_offsets < 1:
            raise ValueError("n_offsets must be >= 1")
        if self.restraint == "s2s" and (self.init is None or self.targ is None):
            raise ValueError("s2s restraint requires init and targ axis labels")
        if self.restraint == "universal" and self.rotation is None:
            raise ValueError("universal restraint requires a rotation (axis, angle_deg)")

    def state_pairs(self) -> list[tuple[ComplexArray, ComplexArray]]:
        """Return (init, targ) operator pairs describing this band's transfers."""
        if self.restraint == "universal":
            assert self.rotation is not None
            rot = _rotation_matrix(*self.rotation)
            return [
                (_bloch_operator(_CARDINAL[axis]), _bloch_operator(rot @ _CARDINAL[axis]))
                for axis in ("x", "y", "z")
            ]
        if self.restraint == "s2s":
            assert self.init is not None and self.targ is not None
            return [
                (_bloch_operator(_axis_bloch(self.init)), _bloch_operator(_axis_bloch(self.targ)))
            ]
        # suppress and xycite both start from Iz; targ is used only by suppress.
        iz = _bloch_operator(_CARDINAL["z"])
        return [(iz, iz)]

    def bloch_pairs(self) -> list[tuple[RealArray, RealArray]]:
        """Return (init, targ) Bloch vectors for the fast kernel path."""
        if self.restraint == "universal":
            assert self.rotation is not None
            rot = _rotation_matrix(*self.rotation)
            return [(_CARDINAL[axis], rot @ _CARDINAL[axis]) for axis in ("x", "y", "z")]
        if self.restraint == "s2s":
            assert self.init is not None and self.targ is not None
            return [(_axis_bloch(self.init), _axis_bloch(self.targ))]
        return [(_CARDINAL["z"], _CARDINAL["z"])]


@dataclass
class SeedlessSpec:
    """A Seedless pulse-design problem for one isolated spin-1/2 channel."""

    spectrometer_mhz: float
    carrier_ppm: float
    rf_max_hz: float
    duration_s: float
    n_steps: int
    bands: list[Band]
    # ponytail: SI Note 2.8 uses 0.95/1.00/1.03 (main-text figures quote
    # 0.93/1.00/1.05). Both are hardware-specific; measure by nutation and pass
    # your own. Defaults follow the formal derivation (SI).
    b1_scales: tuple[float, ...] = (0.95, 1.0, 1.03)
    b1_weights: tuple[float, ...] = (0.25, 0.5, 0.25)
    # fast=True uses the analytic scaled-unitary spin-1/2 kernel; fast=False uses
    # the general 4x4 Liouville GRAPE engine (kept as a cross-check reference).
    fast: bool = True
    _lx: ComplexArray = field(init=False, repr=False)
    _ly: ComplexArray = field(init=False, repr=False)
    _lz: ComplexArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.duration_s <= 0.0 or self.rf_max_hz <= 0.0:
            raise ValueError("duration_s and rf_max_hz must be positive")
        if not self.bands:
            raise ValueError("at least one band is required")
        if len(self.b1_scales) != len(self.b1_weights):
            raise ValueError("b1_scales and b1_weights must have equal length")
        if not math.isclose(sum(self.b1_weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("b1_weights must sum to 1")
        self._lx = liouvillian_comm(Ix())
        self._ly = liouvillian_comm(Iy())
        self._lz = liouvillian_comm(Iz())

    @property
    def dt(self) -> float:
        return self.duration_s / self.n_steps

    def band_offsets_hz(self, band: Band, n: int | None = None) -> RealArray:
        """Return spin offsets (Hz, relative to carrier) spanning a band."""
        points = band.n_offsets if n is None else n
        if points == 1:
            # np.linspace(lo, hi, 1) returns [lo]; sample the band centre instead.
            ppm = np.array([0.5 * (band.ppm_lo + band.ppm_hi)], dtype=np.float64)
        else:
            ppm = np.linspace(band.ppm_lo, band.ppm_hi, points, dtype=np.float64)
        return np.asarray((ppm - self.carrier_ppm) * self.spectrometer_mhz, dtype=np.float64)

    def waveform_xy(self, phases: RealArray) -> RealArray:
        """Return the constant-amplitude XY waveform (fractions of rf_max) for phases."""
        phase = np.asarray(phases, dtype=np.float64)
        return ampl_phase_to_xy(np.ones_like(phase), phase)

    def _control_problem(
        self,
        pairs: list[tuple[ComplexArray, ComplexArray]],
        offsets_hz: RealArray,
        b1_scale: float,
    ) -> ControlProblem:
        """Build a single-B1, phase-only Liouville problem for one band."""
        two_pi_rf = 2.0 * math.pi * b1_scale * self.rf_max_hz
        return ControlProblem(
            drifts=[np.zeros((4, 4), dtype=np.complex128)],
            operators=[two_pi_rf * self._lx, two_pi_rf * self._ly],
            rho_init=[vec(normalise_hs(init)) for init, _ in pairs],
            rho_targ=[vec(normalise_hs(targ)) for _, targ in pairs],
            pulse_dt=self.dt,
            pwr_levels=[1.0, 1.0],
            freeze=None,
            fidelity_mode="real",
            basis="liouville",
            offsets=[float(o) for o in offsets_hz],
            offset_operators=[2.0 * math.pi * self._lz],
        )

    def _member_weights(self, n_offsets: int) -> RealArray:
        """Return per-ensemble-member weights (offset-major, summing to 1)."""
        return np.tile(np.asarray(self.b1_weights, dtype=np.float64), n_offsets) / n_offsets

    def _band_cost(self, band: Band, wfm_xy: RealArray) -> tuple[float, RealArray]:
        """Return this band's infidelity and phase gradient at the given waveform."""
        if self.fast:
            return self._band_cost_fast(band, wfm_xy)
        return self._band_cost_engine(band, wfm_xy)

    def _band_cost_fast(self, band: Band, wfm_xy: RealArray) -> tuple[float, RealArray]:
        """Return the band cost via the analytic scaled-unitary spin-1/2 kernel."""
        offsets = self.band_offsets_hz(band)
        scales = np.asarray(self.b1_scales, dtype=np.float64)
        member_w = self._member_weights(offsets.size)
        z_axis = _CARDINAL["z"]

        if band.restraint == "xycite":
            mz, grad_member = kernel.s2s_value_grad(
                wfm_xy, offsets, scales, self.rf_max_hz, self.dt, z_axis, z_axis
            )
            cost = float(np.sum(member_w * mz * mz))
            grad = np.sum(member_w[:, None] * 2.0 * mz[:, None] * grad_member, axis=0)
            return cost, np.asarray(grad, dtype=np.float64)

        if band.restraint == "suppress" and band.per_step:
            step_cost, grad_member = kernel.suppress_perstep_value_grad(
                wfm_xy, offsets, scales, self.rf_max_hz, self.dt
            )
            cost = float(np.sum(member_w * step_cost))
            grad = np.sum(member_w[:, None] * grad_member, axis=0)
            return cost, np.asarray(grad, dtype=np.float64)

        pairs = band.bloch_pairs()
        cost = 0.0
        grad = np.zeros(self.n_steps, dtype=np.float64)
        for init_vec, targ_vec in pairs:
            fidelity, grad_member = kernel.s2s_value_grad(
                wfm_xy, offsets, scales, self.rf_max_hz, self.dt, init_vec, targ_vec
            )
            cost += float(np.sum(member_w * (1.0 - fidelity))) / len(pairs)
            grad -= np.sum(member_w[:, None] * grad_member, axis=0) / len(pairs)
        return cost, grad

    def _band_cost_engine(self, band: Band, wfm_xy: RealArray) -> tuple[float, RealArray]:
        """Return the band cost via the general Liouville GRAPE engine (reference)."""
        offsets = self.band_offsets_hz(band)
        if band.restraint == "xycite":
            return self._xycite_cost(offsets, wfm_xy)
        if band.restraint == "suppress" and band.per_step:
            raise NotImplementedError("per-step suppression requires the fast kernel (fast=True)")

        pairs = band.state_pairs()
        cost = 0.0
        grad = np.zeros(self.n_steps, dtype=np.float64)
        for scale, weight in zip(self.b1_scales, self.b1_weights):
            cp = self._control_problem(pairs, offsets, scale)
            fidelity, grad_xy = grape_xy_and_gradient(cp, wfm_xy)
            cost += weight * (1.0 - fidelity)
            grad -= weight * phase_only_gradient(grad_xy, wfm_xy)
        return cost, grad

    def _xycite_cost(self, offsets: RealArray, wfm_xy: RealArray) -> tuple[float, RealArray]:
        """Return the squared-residual-Iz cost for an excitation band.

        ponytail: one GRAPE call per (offset, B1) because the squared per-offset
        infidelity cannot be read from the offset-averaged fidelity. Fine at
        verification scale; batch per-offset overlaps if a band grows large.
        """
        iz_pair = [(_bloch_operator(_CARDINAL["z"]), _bloch_operator(_CARDINAL["z"]))]
        cost = 0.0
        grad = np.zeros(self.n_steps, dtype=np.float64)
        n_off = float(offsets.size)
        for scale, weight in zip(self.b1_scales, self.b1_weights):
            for offset in offsets:
                cp = self._control_problem(iz_pair, np.array([offset]), scale)
                mz, grad_xy = grape_xy_and_gradient(cp, wfm_xy)
                grad_phi = phase_only_gradient(grad_xy, wfm_xy)
                cost += weight * mz * mz / n_off
                grad += weight * 2.0 * mz * grad_phi / n_off
        return cost, grad

    def objective(self, phases: RealArray) -> tuple[float, RealArray]:
        """Return total weighted infidelity and its phase gradient."""
        wfm_xy = self.waveform_xy(phases)
        total = 0.0
        grad = np.zeros(self.n_steps, dtype=np.float64)
        for band in self.bands:
            band_cost, band_grad = self._band_cost(band, wfm_xy)
            total += band.weight * band_cost
            grad += band.weight * band_grad
        return total, grad

    def optimize(
        self,
        phases0: RealArray | None = None,
        max_iter: int = 200,
        seed: int | None = 0,
    ) -> tuple[RealArray, float]:
        """Optimise the phase-only waveform; return (phases, final infidelity)."""
        if phases0 is None:
            rng = np.random.default_rng(seed)
            phases0 = rng.uniform(-math.pi, math.pi, size=self.n_steps)
        phases0 = np.asarray(phases0, dtype=np.float64)
        result = minimize(
            self.objective,
            phases0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10},
        )
        return np.asarray(result.x, dtype=np.float64), float(result.fun)

    def evaluate(
        self,
        wfm_xy: RealArray,
        dense: int = 201,
        b1_scale: float = 1.0,
    ) -> dict[str, float]:
        """Return worst-case transfer fidelity per band from the Bloch model.

        The waveform is XY fractions of ``rf_max_hz`` (columns ``[ux, uy]``),
        the same representation :meth:`waveform_xy` produces. ``b1_scale=1.0``
        reproduces on-resonance nominal-B1 evaluation. For ``"xycite"`` the
        reported value is ``max|mz|`` (smaller is better). For a per-step
        ``"suppress"`` band the reported value is the worst ``Iz`` hold across
        *all* prefixes of the pulse, not just its end.
        """
        waveform = np.asarray(wfm_xy, dtype=np.float64)
        scales = np.array([b1_scale], dtype=np.float64)
        results: dict[str, float] = {}
        for index, band in enumerate(self.bands):
            offsets = self.band_offsets_hz(band, n=dense)
            key = f"band{index}:{band.restraint}"
            if band.restraint == "xycite":
                final = propagate_bloch_ensemble(
                    _CARDINAL["z"], waveform, offsets, scales, self.rf_max_hz, self.dt
                )[0]
                results[key] = float(np.max(np.abs(final[:, 2])))
                continue
            if band.restraint == "suppress" and band.per_step:
                worst_hold = 1.0
                for prefix in range(1, waveform.shape[0] + 1):
                    final = propagate_bloch_ensemble(
                        _CARDINAL["z"], waveform[:prefix], offsets, scales, self.rf_max_hz, self.dt
                    )[0]
                    worst_hold = min(worst_hold, float(np.min(final[:, 2])))
                results[key] = worst_hold
                continue
            worst = 1.0
            for init_op, targ_op in band.state_pairs():
                init_vec = _operator_bloch(init_op)
                targ_vec = _operator_bloch(targ_op)
                final = propagate_bloch_ensemble(
                    init_vec, waveform, offsets, scales, self.rf_max_hz, self.dt
                )[0]
                worst = min(worst, float(np.min(final @ targ_vec)))
            results[key] = worst
        return results

    def export_shape(self, phases: RealArray, path: str | Path, title: str = "ocseed") -> Path:
        """Write a Bruker phase-only shape (amplitude 100%, phase in degrees)."""
        phase = np.asarray(phases, dtype=np.float64)
        amplitude = np.full(phase.size, 100.0, dtype=np.float64)
        phase_deg = np.mod(np.degrees(phase), 360.0)
        return export_bruker_shape(
            Path(path),
            title,
            amplitude,
            phase_deg,
            integfac=1.0,
            extra_tags=[
                f"##$OCSEED_DURATION_S= {self.duration_s:.12e}",
                f"##$OCSEED_RF_MAX_HZ= {self.rf_max_hz:.12e}",
                f"##$OCSEED_SPECTROMETER_1H_MHZ= {self.spectrometer_mhz:.12e}",
                f"##$OCSEED_CARRIER_PPM= {self.carrier_ppm:.6f}",
            ],
        )


def _operator_bloch(operator: ComplexArray) -> RealArray:
    """Return the Bloch vector ``[<Ix>,<Iy>,<Iz>]`` scaled to unit length."""
    op = np.asarray(operator, dtype=np.complex128)
    vector = np.array(
        [np.real(np.trace(op @ Ix())), np.real(np.trace(op @ Iy())), np.real(np.trace(op @ Iz()))],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def demo() -> None:
    """Self-check: gradient vs finite difference, and a fast inversion optimise."""
    spec = SeedlessSpec(
        spectrometer_mhz=600.0,
        carrier_ppm=0.0,
        rf_max_hz=10_000.0,
        duration_s=120.0e-6,
        n_steps=40,
        bands=[Band(-8.0, 8.0, "s2s", n_offsets=9, init="-y", targ="y")],
    )
    rng = np.random.default_rng(1)
    phases = rng.uniform(-math.pi, math.pi, size=spec.n_steps)
    _, analytic = spec.objective(phases)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for k in range(spec.n_steps):
        bumped = phases.copy()
        bumped[k] += step
        plus, _ = spec.objective(bumped)
        bumped[k] -= 2.0 * step
        minus, _ = spec.objective(bumped)
        numeric[k] = (plus - minus) / (2.0 * step)
    rel = float(np.max(np.abs(analytic - numeric)) / (np.max(np.abs(numeric)) + 1e-12))
    assert rel < 1e-4, f"gradient mismatch {rel}"

    best_phases, infidelity = spec.optimize(max_iter=150, seed=3)
    assert infidelity < 0.05, f"inversion did not converge: {infidelity}"
    worst = spec.evaluate(spec.waveform_xy(best_phases))["band0:s2s"]
    assert worst > 0.95, f"worst -y->y fidelity {worst}"
    print(f"demo ok (grad rel {rel:.2e}, infidelity {infidelity:.4f}, worst {worst:.4f})")


if __name__ == "__main__":
    demo()
