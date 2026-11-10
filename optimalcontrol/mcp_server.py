"""MCP server exposing the optimalcontrol pulse-design workflows.

Run with ``optimalcontrol-mcp`` (or ``python -m optimalcontrol.mcp_server``);
speaks MCP over stdio. Requires the optional ``mcp`` dependency
(``pip install "optimalcontrol-nmr[mcp]"``).

Large arrays never cross the MCP boundary: design tools write a Bruker
``.shape`` file plus a design JSON and return paths and summary numbers;
analysis tools write CSV/JSON files and return paths and summaries. Units are
the same everywhere: Hz for RF amplitudes and offsets (relative to the
carrier), ppm only for Seedless band edges and carrier, seconds for durations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from optimalcontrol import __version__
from optimalcontrol._seedless_kernel import bloch_operator
from optimalcontrol._types import Array, RealArray
from optimalcontrol._validation import validate_nonnegative, validate_positive
from optimalcontrol.analysis import spectrogram_data
from optimalcontrol.bloch import _rotate, propagate_bloch_ensemble
from optimalcontrol.crop import crop_eta, crop_pulse_params, crop_waveform, crop_zeta
from optimalcontrol.grape import ControlProblem, ampl_phase_to_xy, grape_xy, xy_to_ampl_phase
from optimalcontrol.io import (
    Waveform,
    export_bruker_shape,
    export_csv,
    export_json,
    import_bruker_shape,
    import_csv,
    import_json,
)
from optimalcontrol.ocseed import Band, SeedlessSpec, _axis_bloch, _rotation_matrix
from optimalcontrol.operators import Ix, Iy, Iz, liouvillian_comm, vec
from optimalcontrol.optimizers import run_grape
from optimalcontrol.penalties import PenaltySpec
from optimalcontrol.rope import (
    rope_finite_efficiency,
    rope_g,
    rope_gain_over_inept,
    rope_n,
    rope_switching_time,
    rope_Tcrit,
    rope_waveform,
)
from optimalcontrol.states import normalise_hs, state_from_label

try:
    from mcp.server import MCPServer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'MCP support requires the optional "mcp" dependency: pip install "optimalcontrol-nmr[mcp]"'
    ) from exc

server = MCPServer(
    name="optimalcontrol",
    version=__version__,
    instructions=(
        "Design, verify, and analyse NMR RF pulses with the optimalcontrol package. "
        "Units: RF and offsets in Hz (offsets relative to the carrier), Seedless band "
        "edges and carrier in ppm, durations in seconds. Routes: design_seedless_pulse "
        "(constant-amplitude phase-only, band/restraint input) or design_grape_pulse "
        "(full amplitude+phase GRAPE for one spin-1/2, state labels Ix/Iy/Iz); then "
        "bloch_offset_profile / bloch_trajectory on the returned design_json to verify "
        "with the independent Bloch model. inspect_waveform / convert_waveform read and "
        "convert Bruker .shape, CSV, and JSON pulse files; waveform_spectrogram writes a "
        "time-frequency map; rope_pulse / crop_pulse evaluate the analytic ROPE/CROP "
        "theory. Every tool writes large arrays to files and returns paths plus summaries."
    ),
)

_VERIFY_OFFSETS = 201
_BRUKER_DURATION_TAGS = ("$OCSEED_DURATION_S", "$OPTIMALCONTROL_TOTAL_DURATION_S")
_BRUKER_RF_TAGS = ("$OCSEED_RF_MAX_HZ", "$OPTIMALCONTROL_RF_HZ")


# --------------------------------------------------------------------------- helpers


def _make_spec(spec_dict: dict[str, Any]) -> SeedlessSpec:
    bands = []
    for band in spec_dict["bands"]:
        band = dict(band)
        if band.get("rotation") is not None:
            axis, angle = band["rotation"]
            band["rotation"] = (str(axis), float(angle))
        bands.append(Band(**band))
    kwargs = {k: v for k, v in spec_dict.items() if k != "bands"}
    if "b1_scales" in kwargs:
        kwargs["b1_scales"] = tuple(kwargs["b1_scales"])
    if "b1_weights" in kwargs:
        kwargs["b1_weights"] = tuple(kwargs["b1_weights"])
    return SeedlessSpec(bands=bands, **kwargs)


def _parse_state_label(label: str) -> tuple[float, str]:
    """Return ``(sign, axis)`` for labels like ``"Iz"``, ``"-Iz"``, ``"z"``, ``"-y"``."""
    text = label.strip()
    sign = 1.0
    if text and text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    if len(text) == 2 and text[0] == "I":
        text = text[1]
    if text.lower() not in ("x", "y", "z"):
        raise ValueError(
            f"state label must be one of Ix, Iy, Iz or x, y, z (optionally signed), got {label!r}"
        )
    return sign, text.lower()


def _bloch_vector(label: str) -> RealArray:
    sign, axis = _parse_state_label(label)
    return _axis_bloch(("-" if sign < 0 else "") + axis)


def _density_state(label: str) -> Array:
    sign, axis = _parse_state_label(label)
    return np.asarray(np.complex128(sign) * state_from_label("I" + axis, 1), dtype=np.complex128)


def _offset_grid(offset_lo_hz: float, offset_hi_hz: float, n_offsets: int) -> RealArray:
    """Return ``n_offsets`` Hz offsets spanning the range (the centre when n is 1)."""
    for name, value in (("offset_lo_hz", offset_lo_hz), ("offset_hi_hz", offset_hi_hz)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if offset_hi_hz < offset_lo_hz:
        raise ValueError("offset_hi_hz must be >= offset_lo_hz")
    if n_offsets < 1:
        raise ValueError("n_offsets must be >= 1")
    if n_offsets == 1:
        return np.array([0.5 * (offset_lo_hz + offset_hi_hz)], dtype=np.float64)
    return np.linspace(offset_lo_hz, offset_hi_hz, n_offsets, dtype=np.float64)


def _validated_scales(name: str, scales: list[float] | None) -> list[float]:
    if scales is None:
        return [1.0]
    if not scales:
        raise ValueError(f"{name} must be non-empty")
    for index, scale in enumerate(scales):
        validate_positive(f"{name}[{index}]", float(scale))
    return [float(scale) for scale in scales]


def _output_dir(out_dir: str) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _output_file(out_path: str) -> Path:
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _amplitude_phase_deg(waveform_xy: RealArray) -> tuple[RealArray, RealArray]:
    """Return ``(amplitude_percent, phase_deg in [0, 360))`` for Bruker export."""
    amplitude, phase = xy_to_ampl_phase(waveform_xy)
    phase_deg = np.mod(np.degrees(phase), 360.0)
    phase_deg[amplitude < 1e-12] = 0.0
    return 100.0 * amplitude, np.asarray(phase_deg, dtype=np.float64)


def _write_shape(
    path: Path,
    title: str,
    waveform_xy: RealArray,
    duration_s: float | None,
    rf_max_hz: float | None,
) -> Path:
    """Write a Bruker shape-library file (amplitude %, phase deg) with provenance tags."""
    amplitude_percent, phase_deg = _amplitude_phase_deg(waveform_xy)
    constant = bool(np.allclose(amplitude_percent, amplitude_percent[0], rtol=0.0, atol=1e-9))
    tags: list[str] = []
    if duration_s is not None:
        tags.append(f"##$OPTIMALCONTROL_TOTAL_DURATION_S= {duration_s:.12e}")
        tags.append(f"##$OPTIMALCONTROL_STEP_DURATION_S= {duration_s / waveform_xy.shape[0]:.12e}")
    if rf_max_hz is not None:
        tags.append(f"##$OPTIMALCONTROL_RF_HZ= {rf_max_hz:.12e}")
    tags.append(
        "##$OPTIMALCONTROL_NOTE= Set pulse length to TOTAL_DURATION_S "
        "and calibrate 100% to RF_HZ."
    )
    return export_bruker_shape(
        path,
        title,
        amplitude_percent,
        phase_deg,
        shape_mode=1 if constant else 0,
        extra_tags=tags,
    )


@dataclass(frozen=True)
class _Pulse:
    """A pulse loaded from any supported file, as XY fractions of the RF maximum."""

    waveform_xy: RealArray
    rf_max_hz: float | None
    duration_s: float | None
    source: str

    @property
    def n_steps(self) -> int:
        return int(self.waveform_xy.shape[0])

    @property
    def dt_s(self) -> float | None:
        return None if self.duration_s is None else self.duration_s / self.n_steps

    def require_dt(self) -> float:
        if self.duration_s is None:
            raise ValueError(
                f"duration_s is required: the {self.source} file carries no pulse duration"
            )
        return self.duration_s / self.n_steps

    def require_rf(self) -> float:
        if self.rf_max_hz is None:
            raise ValueError(
                f"rf_max_hz is required: the {self.source} file carries no RF calibration"
            )
        return self.rf_max_hz


def _xy_from_waveform(wfm: Waveform) -> RealArray:
    """Return ``(n_steps, 2)`` XY data from a two-channel :class:`Waveform`."""
    if len(wfm.channels) != 2:
        raise ValueError(
            f"waveform file must have exactly 2 channels (x, y), got {len(wfm.channels)}"
        )
    lowered = [channel.lower() for channel in wfm.channels]
    if "x" in lowered and "y" in lowered:
        order = (lowered.index("x"), lowered.index("y"))
    else:
        order = (0, 1)
    return np.asarray(wfm.data[list(order), :].T, dtype=np.float64)


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _waveform_duration(wfm: Waveform) -> float | None:
    times = wfm.times
    if times.shape[0] >= 2:
        dt: float | None = float(times[1] - times[0])
    else:
        dt = _metadata_float(wfm.metadata, "pulse_dt")
    if dt is None or dt <= 0.0:
        return None
    return float(dt * times.shape[0])


def _tag_float(tags: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in tags:
            return float(tags[key])
    return None


def _load_design_json(payload: dict[str, Any], source_path: Path) -> _Pulse:
    """Return the pulse stored in a design JSON written by the design tools."""
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"design JSON {source_path} is missing the 'spec' object")
    if "phases_rad" in payload:
        seedless = _make_spec(spec)
        phases = np.asarray(payload["phases_rad"], dtype=np.float64)
        return _Pulse(
            seedless.waveform_xy(phases), seedless.rf_max_hz, seedless.duration_s, "seedless"
        )
    if "waveform_xy" in payload:
        waveform = np.asarray(payload["waveform_xy"], dtype=np.float64)
        if waveform.ndim != 2 or waveform.shape[1] != 2 or waveform.shape[0] == 0:
            raise ValueError(
                f"design JSON waveform_xy must have shape (n_steps, 2), got {waveform.shape}"
            )
        if not np.all(np.isfinite(waveform)):
            raise ValueError("design JSON waveform_xy entries must be finite")
        return _Pulse(waveform, float(spec["rf_max_hz"]), float(spec["duration_s"]), "grape")
    raise ValueError(
        f"design JSON {source_path} must contain 'phases_rad' (Seedless) or 'waveform_xy' (GRAPE)"
    )


def _load_pulse(
    path: str, duration_s: float | None = None, rf_max_hz: float | None = None
) -> _Pulse:
    """Load a design JSON, waveform JSON/CSV, or Bruker ``.shape`` file as XY fractions."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"file not found: {file}")
    suffix = file.suffix.lower()
    if suffix == ".json":
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{file} is not a JSON object")
        if "spec" in payload or "phases_rad" in payload or "waveform_xy" in payload:
            pulse = _load_design_json(payload, file)
        elif "channels" in payload and "data" in payload:
            wfm = import_json(file)
            pulse = _Pulse(
                _xy_from_waveform(wfm),
                _metadata_float(wfm.metadata, "rf_max_hz"),
                _waveform_duration(wfm),
                "json",
            )
        else:
            raise ValueError(f"{file} is neither a design JSON nor a waveform JSON")
    elif suffix == ".csv":
        wfm = import_csv(file)
        pulse = _Pulse(_xy_from_waveform(wfm), None, _waveform_duration(wfm), "csv")
    else:
        amplitude_percent, phase_deg, tags = import_bruker_shape(file)
        waveform = ampl_phase_to_xy(amplitude_percent / 100.0, np.radians(phase_deg))
        pulse = _Pulse(
            waveform,
            _tag_float(tags, _BRUKER_RF_TAGS),
            _tag_float(tags, _BRUKER_DURATION_TAGS),
            "bruker",
        )
    if duration_s is not None:
        validate_positive("duration_s", duration_s)
        pulse = _Pulse(pulse.waveform_xy, pulse.rf_max_hz, float(duration_s), pulse.source)
    if rf_max_hz is not None:
        validate_positive("rf_max_hz", rf_max_hz)
        pulse = _Pulse(pulse.waveform_xy, float(rf_max_hz), pulse.duration_s, pulse.source)
    return pulse


def _io_waveform(pulse: _Pulse) -> Waveform:
    """Return an ``x``/``y`` :class:`Waveform` (fractions of rf_max) with a real time axis."""
    dt = pulse.require_dt()
    metadata: dict[str, object] = {
        "optimalcontrol_version": __version__,
        "source": pulse.source,
        "pulse_dt": dt,
        "duration_s": pulse.duration_s,
    }
    if pulse.rf_max_hz is not None:
        metadata["rf_max_hz"] = pulse.rf_max_hz
    return Waveform(
        channels=["x", "y"],
        units="fraction of rf_max_hz",
        times=np.arange(pulse.n_steps, dtype=np.float64) * dt,
        data=np.asarray(pulse.waveform_xy.T, dtype=np.float64),
        metadata=metadata,
        problem_hash=f"mcp:{pulse.source}",
    )


def _grape_problem(
    pairs: list[tuple[Array, Array]],
    offsets_hz: RealArray,
    b1_scales: list[float],
    rf_max_hz: float,
    dt: float,
    amplitude_penalty_weight: float,
) -> ControlProblem:
    """Build the single-spin Liouville GRAPE problem over an offset x B1 ensemble.

    ``pwr_levels`` is a B1 ensemble axis only when its length differs from the
    operator count, so a single scale is applied per channel instead.
    """
    two_pi_rf = 2.0 * math.pi * rf_max_hz
    pwr_levels = [b1_scales[0], b1_scales[0]] if len(b1_scales) == 1 else list(b1_scales)
    return ControlProblem(
        drifts=[np.zeros((4, 4), dtype=np.complex128)],
        operators=[two_pi_rf * liouvillian_comm(Ix()), two_pi_rf * liouvillian_comm(Iy())],
        rho_init=[vec(normalise_hs(init)) for init, _ in pairs],
        rho_targ=[vec(normalise_hs(targ)) for _, targ in pairs],
        pulse_dt=dt,
        pwr_levels=pwr_levels,
        freeze=None,
        fidelity_mode="real",
        basis="liouville",
        offsets=[float(offset) for offset in offsets_hz],
        offset_operators=[2.0 * math.pi * liouvillian_comm(Iz())],
        penalties=(
            [PenaltySpec(kind="SNSA", weight=amplitude_penalty_weight, limit=1.0)]
            if amplitude_penalty_weight > 0.0
            else None
        ),
    )


def _bloch_transfer(
    pairs: list[tuple[RealArray, RealArray]],
    waveform_xy: RealArray,
    offsets_hz: RealArray,
    b1_scales: RealArray,
    rf_max_hz: float,
    dt: float,
) -> RealArray:
    """Return pair-averaged Bloch transfer ``<targ|final>`` shaped ``(n_b1, n_offsets)``."""
    total = np.zeros((b1_scales.size, offsets_hz.size), dtype=np.float64)
    for init_vec, targ_vec in pairs:
        final = propagate_bloch_ensemble(
            init_vec, waveform_xy, offsets_hz, b1_scales, rf_max_hz, dt
        )
        total += final @ targ_vec
    return np.asarray(total / len(pairs), dtype=np.float64)


def _transverse_rotation(pairs: list[tuple[RealArray, RealArray]]) -> tuple[str, float] | None:
    """Return ``(axis, angle_deg)`` when three pairs spell a rotation about x or y."""
    if len(pairs) != 3:
        return None
    init = np.array([a for a, _ in pairs])
    rot = np.array([b for _, b in pairs]).T @ init
    eye = np.eye(3)
    if not (np.allclose(init @ init.T, eye) and np.allclose(rot @ rot.T, eye)):
        return None
    if np.linalg.det(rot) < 0.0:
        return None
    for axis, k, cos, sin in (("x", 0, rot[1, 1], rot[2, 1]), ("y", 1, rot[2, 2], rot[0, 2])):
        if np.isclose(rot[k, k], 1.0) and not np.isclose(cos, 1.0):
            return axis, math.degrees(math.atan2(sin, cos))
    return None


# --------------------------------------------------------------------------- tools


@server.tool()
def design_seedless_pulse(
    spectrometer_mhz: float,
    carrier_ppm: float,
    rf_max_hz: float,
    duration_s: float,
    n_steps: int,
    bands: list[dict[str, Any]],
    out_dir: str,
    name: str = "pulse",
    n_seeds: int = 4,
    max_iter: int = 300,
    b1_scales: list[float] | None = None,
    b1_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Design a constant-amplitude phase-only pulse via the Seedless route.

    Each entry of ``bands`` is a dict with keys ``ppm_lo``, ``ppm_hi``,
    ``restraint`` and optionally ``n_offsets``, ``weight``, and per-restraint
    fields: ``"s2s"`` needs ``init``/``targ`` axis labels (e.g. "-y", "y");
    ``"universal"`` needs ``rotation`` as [axis, angle_deg] (e.g. ["x", 180.0]);
    ``"suppress"`` (a water hold) optionally takes ``per_step``; ``"xycite"``
    needs nothing extra.

    Runs ``n_seeds`` random starts and keeps the best (the cost is non-convex;
    universal rotations about x or y start from a mirrored half-length pulse
    rather than raw random phases, which is far more reliable),
    verifies the winner on a dense offset grid across all ``b1_scales`` with an
    independent Bloch model, and writes ``<out_dir>/<name>.shape`` (Bruker) plus
    ``<out_dir>/<name>.json`` (spec + phases, input for bloch_offset_profile).

    The returned ``worst_case`` dict takes the worst value over the full B1
    ensemble, regardless of ``b1_weights``. 1.0 is perfect — except ``xycite``
    entries, which report maximum leftover |mz| where 0.0 is perfect. A negative
    value means a wrong-sign local minimum: retry with more n_seeds, or give
    the problem more duration_s / n_steps / rf_max_hz.
    """
    spec_dict: dict[str, Any] = {
        "spectrometer_mhz": spectrometer_mhz,
        "carrier_ppm": carrier_ppm,
        "rf_max_hz": rf_max_hz,
        "duration_s": duration_s,
        "n_steps": n_steps,
        "bands": bands,
    }
    if b1_scales is not None:
        spec_dict["b1_scales"] = b1_scales
    if b1_weights is not None:
        spec_dict["b1_weights"] = b1_weights
    spec = _make_spec(spec_dict)

    best_phases, best_infidelity = None, float("inf")
    for seed in range(n_seeds):
        phases, infidelity = spec.optimize(max_iter=max_iter, seed=seed)
        if infidelity < best_infidelity:
            best_phases, best_infidelity = phases, infidelity
    assert best_phases is not None

    waveform = spec.waveform_xy(best_phases)
    evaluations = [spec.evaluate(waveform, b1_scale=scale) for scale in spec.b1_scales]
    worst_case = {
        key: (max if key.endswith(":xycite") else min)(values[key] for values in evaluations)
        for key in evaluations[0]
    }

    out = _output_dir(out_dir)
    shape_path = spec.export_shape(best_phases, out / f"{name}.shape", title=name)
    design_path = out / f"{name}.json"
    design_path.write_text(
        json.dumps(
            {
                "optimalcontrol_version": __version__,
                "spec": spec_dict,
                "phases_rad": best_phases.tolist(),
                "infidelity": best_infidelity,
                "worst_case": worst_case,
            },
            indent=2,
        )
    )

    warnings = [
        f"{key} = {value:.4f}: wrong-sign local minimum — see docstring"
        for key, value in worst_case.items()
        if value < 0.0 and not key.endswith(":xycite")
    ]
    return {
        "shape_file": str(shape_path),
        "design_json": str(design_path),
        "infidelity": round(best_infidelity, 6),
        "worst_case": {k: round(v, 4) for k, v in worst_case.items()},
        "dt_s": spec.dt,
        "warnings": warnings,
    }


@server.tool()
def design_grape_pulse(
    init: list[str],
    targ: list[str],
    rf_max_hz: float,
    duration_s: float,
    n_steps: int,
    offset_lo_hz: float,
    offset_hi_hz: float,
    out_dir: str,
    n_offsets: int = 11,
    b1_scales: list[float] | None = None,
    name: str = "grape",
    method: str = "lbfgs",
    seed: int = 0,
    max_iter: int = 200,
    amplitude_penalty_weight: float = 0.05,
) -> dict[str, Any]:
    """Design an amplitude+phase pulse for an isolated spin-1/2 with full GRAPE.

    ``init`` and ``targ`` are equal-length lists of product-operator labels
    ``Ix``/``Iy``/``Iz`` with an optional sign, paired index by index. One pair
    is a state-to-state transfer (``init=["Iz"], targ=["-Iz"]`` is an
    inversion); three pairs make a universal rotation (a 180x UR is
    ``init=["Ix", "Iy", "Iz"], targ=["Ix", "-Iy", "-Iz"]``); the optimiser
    maximises the mean real fidelity over all pairs. Two-spin problems
    (``2IzSz`` labels, J coupling, relaxation) are not exposed here — build a
    ControlProblem in Python for those (docs/guide_grape.md).

    Offsets are Hz relative to the carrier: ``n_offsets`` points spanning
    ``offset_lo_hz..offset_hi_hz`` (convert ppm with
    ``offset_hz = (ppm - carrier_ppm) * spectrometer_mhz``). ``b1_scales`` are
    RF-inhomogeneity multipliers (e.g. [0.9, 1.0, 1.1]; the GRAPE engine reads
    exactly two entries as per-channel powers, so give 1 or at least 3).
    ``rf_max_hz`` caps the RF amplitude (nutation frequency, Hz): the
    optimiser penalises amplitude above the cap with weight
    ``amplitude_penalty_weight`` (SNSA spillout penalty) and the final
    waveform is hard-clipped to the cap before verification and export.
    ``method`` is "lbfgs" or "newton"; ``seed`` picks the random small-
    amplitude starting waveform. Three pairs forming a rotation about x or y
    (with even ``n_steps``) instead start from a symmetric construction: a
    half-length pulse is optimised first and mirrored into a full-length
    start, which avoids the wrong-sign minima random starts fall into.

    Writes ``<out_dir>/<name>.shape`` (Bruker, variable amplitude in percent of
    rf_max_hz, phase in degrees) and ``<out_dir>/<name>.json`` (spec +
    waveform_xy as fractions of rf_max_hz, input for bloch_offset_profile,
    bloch_trajectory, convert_waveform). Returns ``fidelity`` (GRAPE ensemble
    mean on the design grid, 1.0 perfect), ``bloch_mean`` (same grid, the
    independent Bloch model — should agree), ``bloch_worst`` (minimum transfer
    over a dense 201-point offset grid x all b1_scales, with its location),
    optimiser iterations/convergence, and clipping statistics.
    """
    if not init or not targ:
        raise ValueError("init and targ must be non-empty lists of state labels")
    if len(init) != len(targ):
        raise ValueError(f"init and targ must have equal length, got {len(init)} and {len(targ)}")
    validate_positive("rf_max_hz", rf_max_hz)
    validate_positive("duration_s", duration_s)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    validate_nonnegative("amplitude_penalty_weight", amplitude_penalty_weight)
    scales = _validated_scales("b1_scales", b1_scales)
    if len(scales) == 2:
        raise ValueError(
            "b1_scales must have 1 or at least 3 entries "
            "(the GRAPE engine reads exactly 2 entries as per-channel power levels)"
        )
    offsets = _offset_grid(offset_lo_hz, offset_hi_hz, n_offsets)
    density_pairs = [(_density_state(a), _density_state(b)) for a, b in zip(init, targ)]
    bloch_pairs = [(_bloch_vector(a), _bloch_vector(b)) for a, b in zip(init, targ)]
    dt = duration_s / n_steps

    cp = _grape_problem(density_pairs, offsets, scales, rf_max_hz, dt, amplitude_penalty_weight)
    rng = np.random.default_rng(seed)
    wfm0 = np.asarray(0.1 * rng.standard_normal((n_steps, 2)), dtype=np.float64)
    rotation = _transverse_rotation(bloch_pairs)
    if rotation is not None and n_steps % 2 == 0:
        # Random starts trap universal rotations in wrong-sign minima; start from
        # the symmetric construction instead (see SeedlessSpec._symmetric_start).
        axis, angle_deg = rotation
        start = _bloch_vector("y" if axis == "x" else "x")
        half_targ = _rotation_matrix(axis, angle_deg / 2.0) @ start
        half_cp = _grape_problem(
            [(bloch_operator(start), bloch_operator(half_targ))],
            offsets,
            scales,
            rf_max_hz,
            dt,
            amplitude_penalty_weight,
        )
        _, half = run_grape(half_cp, wfm0[: n_steps // 2], method=method, max_iter=max_iter)
        half_wfm = np.asarray(half.wfm_final, dtype=np.float64)
        mirror = np.array([1.0, -1.0] if axis == "x" else [-1.0, 1.0])
        wfm0 = np.concatenate([half_wfm[::-1] * mirror, half_wfm])
    _, result = run_grape(cp, wfm0, method=method, max_iter=max_iter)

    amplitude, phase = xy_to_ampl_phase(np.asarray(result.wfm_final, dtype=np.float64))
    max_amplitude = float(np.max(amplitude))
    clipped_steps = int(np.count_nonzero(amplitude > 1.0))
    waveform = ampl_phase_to_xy(np.minimum(amplitude, 1.0), phase)
    fidelity = grape_xy(cp, waveform)

    scale_array = np.asarray(scales, dtype=np.float64)
    bloch_mean = float(
        np.mean(_bloch_transfer(bloch_pairs, waveform, offsets, scale_array, rf_max_hz, dt))
    )
    dense = _offset_grid(offset_lo_hz, offset_hi_hz, _VERIFY_OFFSETS)
    dense_transfer = _bloch_transfer(bloch_pairs, waveform, dense, scale_array, rf_max_hz, dt)
    worst_b1, worst_offset = np.unravel_index(np.argmin(dense_transfer), dense_transfer.shape)
    bloch_worst = float(dense_transfer[worst_b1, worst_offset])

    spec_dict: dict[str, Any] = {
        "init": list(init),
        "targ": list(targ),
        "rf_max_hz": rf_max_hz,
        "duration_s": duration_s,
        "n_steps": n_steps,
        "offset_lo_hz": offset_lo_hz,
        "offset_hi_hz": offset_hi_hz,
        "n_offsets": n_offsets,
        "b1_scales": scales,
        "method": method,
        "seed": seed,
        "max_iter": max_iter,
        "amplitude_penalty_weight": amplitude_penalty_weight,
    }
    out = _output_dir(out_dir)
    shape_path = _write_shape(out / f"{name}.shape", name, waveform, duration_s, rf_max_hz)
    design_path = out / f"{name}.json"
    design_path.write_text(
        json.dumps(
            {
                "optimalcontrol_version": __version__,
                "kind": "grape",
                "spec": spec_dict,
                "waveform_xy": waveform.tolist(),
                "fidelity": fidelity,
                "bloch_mean": bloch_mean,
                "bloch_worst": bloch_worst,
                "bloch_worst_offset_hz": float(dense[worst_offset]),
                "bloch_worst_b1_scale": float(scale_array[worst_b1]),
                "optimizer": {
                    "fidelity_final": result.fidelity_final,
                    "n_iter": result.n_iter,
                    "n_feval": result.n_feval,
                    "converged": result.converged,
                    "reason": result.reason,
                },
                "max_amplitude_before_clip": max_amplitude,
                "clipped_steps": clipped_steps,
            },
            indent=2,
        )
    )
    warnings: list[str] = []
    if clipped_steps:
        warnings.append(
            f"{clipped_steps} step(s) exceeded rf_max_hz (max {max_amplitude:.3f} x) and were "
            "clipped; raise amplitude_penalty_weight, duration_s or rf_max_hz if fidelity suffered"
        )
    if bloch_worst < 0.0:
        warnings.append(
            f"bloch_worst = {bloch_worst:.4f}: wrong-sign region — retry with another seed "
            "or more duration_s / n_steps / rf_max_hz"
        )
    return {
        "shape_file": str(shape_path),
        "design_json": str(design_path),
        "fidelity": round(fidelity, 6),
        "bloch_mean": round(bloch_mean, 6),
        "bloch_worst": round(bloch_worst, 6),
        "bloch_worst_offset_hz": round(float(dense[worst_offset]), 3),
        "bloch_worst_b1_scale": float(scale_array[worst_b1]),
        "n_iter": result.n_iter,
        "n_feval": result.n_feval,
        "converged": result.converged,
        "reason": result.reason,
        "max_amplitude_before_clip": round(max_amplitude, 4),
        "clipped_steps": clipped_steps,
        "dt_s": dt,
        "warnings": warnings,
    }


@server.tool()
def bloch_offset_profile(
    design_json: str,
    offset_lo_hz: float,
    offset_hi_hz: float,
    n_points: int = 101,
    init: str = "z",
    b1_scale: float = 1.0,
    targ: str | None = None,
    b1_scales: list[float] | None = None,
) -> dict[str, Any]:
    """Propagate a designed pulse across offsets with the independent Bloch model.

    ``design_json`` is the file written by design_seedless_pulse or
    design_grape_pulse. Offsets are Hz relative to the carrier. ``init`` is the
    starting state ("x", "y", "z" or "Ix", "Iy", "Iz", optionally negated).
    Returns the final magnetisation per offset at ``b1_scale``; regions not
    covered by any band / the design offset range are unconstrained by design.

    Optional ``targ`` (same label syntax) adds ``transfer`` — the projection
    of the final state on the target per offset (1.0 perfect) — and a
    ``summary`` with the worst and mean transfer. Optional ``b1_scales``
    extends the summary to the worst case over that whole offset x B1 grid
    (``mx``/``my``/``mz``/``transfer`` stay at ``b1_scale``).
    """
    pulse = _load_pulse(design_json)
    validate_positive("b1_scale", b1_scale)
    offsets = _offset_grid(offset_lo_hz, offset_hi_hz, n_points)
    init_vec = _bloch_vector(init)
    rf_max_hz, dt = pulse.require_rf(), pulse.require_dt()
    final = propagate_bloch_ensemble(
        init_vec, pulse.waveform_xy, offsets, np.array([b1_scale]), rf_max_hz, dt
    )[0]
    response: dict[str, Any] = {
        "offsets_hz": [round(float(x), 3) for x in offsets],
        "mx": [round(float(x), 4) for x in final[:, 0]],
        "my": [round(float(x), 4) for x in final[:, 1]],
        "mz": [round(float(x), 4) for x in final[:, 2]],
    }
    if targ is None:
        if b1_scales is not None:
            raise ValueError("b1_scales requires targ")
        return response
    targ_vec = _bloch_vector(targ)
    transfer = final @ targ_vec
    response["transfer"] = [round(float(x), 4) for x in transfer]
    if b1_scales is None:
        scale_array = np.array([b1_scale], dtype=np.float64)
    else:
        scale_array = np.asarray(_validated_scales("b1_scales", b1_scales), dtype=np.float64)
    grid = _bloch_transfer(
        [(init_vec, targ_vec)], pulse.waveform_xy, offsets, scale_array, rf_max_hz, dt
    )
    worst_b1, worst_offset = np.unravel_index(np.argmin(grid), grid.shape)
    response["summary"] = {
        "worst_transfer": round(float(grid[worst_b1, worst_offset]), 6),
        "worst_offset_hz": round(float(offsets[worst_offset]), 3),
        "worst_b1_scale": float(scale_array[worst_b1]),
        "mean_transfer": round(float(np.mean(grid)), 6),
        "b1_scales": [float(x) for x in scale_array],
    }
    return response


@server.tool()
def inspect_waveform(
    path: str,
    duration_s: float | None = None,
    rf_max_hz: float | None = None,
) -> dict[str, Any]:
    """Summarise a pulse file without returning the waveform itself.

    Accepts a design JSON from the design tools, a waveform JSON/CSV as written
    by convert_waveform (or ``optimalcontrol.io``, channels x/y), or a Bruker
    shape-library ``.shape``/JCAMP file (amplitude in percent, phase in
    degrees). Duration and RF calibration are taken from the file when it
    carries them (design JSONs always do; shapes written by this package carry
    ``##$..._DURATION_S`` / ``..._RF_HZ`` tags); pass ``duration_s`` (s) and
    ``rf_max_hz`` (Hz) to supply or override them. Amplitudes are reported as
    fractions of rf_max_hz; phases in degrees.
    """
    pulse = _load_pulse(path, duration_s, rf_max_hz)
    amplitude, phase = xy_to_ampl_phase(pulse.waveform_xy)
    phase_deg = np.mod(np.degrees(phase), 360.0)
    phase_deg[amplitude < 1e-12] = 0.0
    distinct = np.unique(np.round(phase_deg, 6))
    max_amplitude = float(np.max(amplitude))
    return {
        "path": str(Path(path).expanduser()),
        "format": pulse.source,
        "n_steps": pulse.n_steps,
        "duration_s": pulse.duration_s,
        "dt_s": pulse.dt_s,
        "rf_max_hz": pulse.rf_max_hz,
        "max_amplitude": round(max_amplitude, 6),
        "min_amplitude": round(float(np.min(amplitude)), 6),
        "mean_amplitude": round(float(np.mean(amplitude)), 6),
        "constant_amplitude": bool(np.allclose(amplitude, amplitude[0], rtol=0.0, atol=1e-9)),
        "max_rf_hz": None if pulse.rf_max_hz is None else round(max_amplitude * pulse.rf_max_hz, 3),
        "phase_min_deg": round(float(np.min(phase_deg)), 4),
        "phase_max_deg": round(float(np.max(phase_deg)), 4),
        "n_distinct_phases": int(distinct.size),
        "binary_phase": bool(distinct.size <= 2),
    }


@server.tool()
def convert_waveform(
    path: str,
    out_path: str,
    format: str,
    duration_s: float | None = None,
    rf_max_hz: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Convert a pulse file (design JSON, waveform JSON/CSV, Bruker shape) to another format.

    ``format`` is "bruker" (shape-library JCAMP: amplitude percent of
    rf_max_hz, phase degrees, duration/RF provenance tags when known), "csv"
    (columns time,x,y as fractions of rf_max_hz) or "json" (the
    ``optimalcontrol.io`` waveform JSON with channels x/y). csv and json need
    a known pulse duration (from the file or ``duration_s``); the same input
    rules as inspect_waveform apply. ``title`` names the Bruker shape
    (defaults to the output file stem).
    """
    kind = format.strip().lower()
    if kind not in ("bruker", "csv", "json"):
        raise ValueError(f"format must be one of bruker, csv, json, got {format!r}")
    pulse = _load_pulse(path, duration_s, rf_max_hz)
    out = _output_file(out_path)
    if kind == "bruker":
        _write_shape(out, title or out.stem, pulse.waveform_xy, pulse.duration_s, pulse.rf_max_hz)
    else:
        wfm = _io_waveform(pulse)
        if kind == "csv":
            export_csv(wfm, out)
        else:
            export_json(wfm, out)
    return {
        "out_path": str(out),
        "format": kind,
        "source_format": pulse.source,
        "n_steps": pulse.n_steps,
        "duration_s": pulse.duration_s,
        "rf_max_hz": pulse.rf_max_hz,
    }


@server.tool()
def bloch_trajectory(
    path: str,
    offsets_hz: list[float],
    out_path: str,
    init: str = "z",
    b1_scale: float = 1.0,
    duration_s: float | None = None,
    rf_max_hz: float | None = None,
) -> dict[str, Any]:
    """Record the Bloch-vector time course of a pulse at given offsets.

    ``path`` accepts the same files as inspect_waveform (a design JSON needs
    nothing else; Bruker/CSV/JSON files need a known duration and RF
    calibration, via tags or ``duration_s``/``rf_max_hz``). ``offsets_hz`` are
    Hz relative to the carrier; ``init`` is the starting state (x/y/z or
    Ix/Iy/Iz, optionally negated); ``b1_scale`` scales the RF amplitude.

    Writes the trajectory after every step to ``out_path``: a ``.csv`` with
    columns step,time_s,offset_hz,mx,my,mz (one row per step per offset) or a
    ``.json`` with time_s, offsets_hz and mx/my/mz arrays shaped
    [n_steps+1][n_offsets]. Returns the path plus the final Bloch vector per
    offset.
    """
    if not offsets_hz:
        raise ValueError("offsets_hz must be non-empty")
    for index, value in enumerate(offsets_hz):
        if not math.isfinite(value):
            raise ValueError(f"offsets_hz[{index}] must be finite")
    validate_positive("b1_scale", b1_scale)
    pulse = _load_pulse(path, duration_s, rf_max_hz)
    rf, dt = pulse.require_rf(), pulse.require_dt()
    out = _output_file(out_path)
    suffix = out.suffix.lower()
    if suffix not in (".csv", ".json"):
        raise ValueError(f"out_path must end in .csv or .json, got {out.name!r}")

    offsets = np.asarray(offsets_hz, dtype=np.float64)
    state = np.tile(_bloch_vector(init), (offsets.size, 1))
    field = np.empty((offsets.size, 3), dtype=np.float64)
    field[:, 2] = offsets
    trajectory = np.empty((pulse.n_steps + 1, offsets.size, 3), dtype=np.float64)
    trajectory[0] = state
    for step, xy in enumerate(pulse.waveform_xy, start=1):
        field[:, :2] = b1_scale * rf * xy
        state = _rotate(state, field, dt)
        trajectory[step] = state
    time_s = np.arange(pulse.n_steps + 1, dtype=np.float64) * dt

    if suffix == ".csv":
        lines = ["step,time_s,offset_hz,mx,my,mz"]
        for step in range(pulse.n_steps + 1):
            for col, offset in enumerate(offsets):
                mx, my, mz = trajectory[step, col]
                lines.append(f"{step},{time_s[step]:.12e},{offset:.9g},{mx:.9e},{my:.9e},{mz:.9e}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        out.write_text(
            json.dumps(
                {
                    "time_s": time_s.tolist(),
                    "offsets_hz": offsets.tolist(),
                    "init": init,
                    "b1_scale": b1_scale,
                    "mx": trajectory[:, :, 0].tolist(),
                    "my": trajectory[:, :, 1].tolist(),
                    "mz": trajectory[:, :, 2].tolist(),
                }
            ),
            encoding="utf-8",
        )
    return {
        "out_path": str(out),
        "n_steps": pulse.n_steps,
        "n_offsets": int(offsets.size),
        "dt_s": dt,
        "final": [
            {
                "offset_hz": float(offset),
                "mx": round(float(trajectory[-1, col, 0]), 4),
                "my": round(float(trajectory[-1, col, 1]), 4),
                "mz": round(float(trajectory[-1, col, 2]), 4),
            }
            for col, offset in enumerate(offsets)
        ],
    }


@server.tool()
def waveform_spectrogram(
    path: str,
    out_path: str,
    duration_s: float | None = None,
    rf_max_hz: float | None = None,
) -> dict[str, Any]:
    """Write the power spectrogram of a pulse's complex envelope x + i*y.

    ``path`` accepts the same files as inspect_waveform; the pulse duration
    must be known (file tags or ``duration_s``). When rf_max_hz is known the
    envelope is scaled to Hz before the transform, otherwise it stays in
    fractions of rf_max. Writes a JSON file with ``times_s`` (segment centres),
    ``freqs_hz`` (two-sided, in Hz relative to the carrier) and ``power``
    shaped [n_freqs][n_times]; returns the path, grid sizes and the strongest
    time-frequency bin.
    """
    pulse = _load_pulse(path, duration_s, rf_max_hz)
    dt = pulse.require_dt()
    waveform = pulse.waveform_xy if pulse.rf_max_hz is None else pulse.rf_max_hz * pulse.waveform_xy
    times, freqs, power = spectrogram_data(waveform, (0, 1), dt)
    out = _output_file(out_path)
    out.write_text(
        json.dumps(
            {
                "times_s": times.tolist(),
                "freqs_hz": freqs.tolist(),
                "power": power.tolist(),
                "envelope_units": "fraction of rf_max" if pulse.rf_max_hz is None else "Hz",
            }
        ),
        encoding="utf-8",
    )
    peak_f, peak_t = np.unravel_index(np.argmax(power), power.shape)
    return {
        "out_path": str(out),
        "n_times": int(times.size),
        "n_freqs": int(freqs.size),
        "freq_resolution_hz": round(float(freqs[1] - freqs[0]), 3) if freqs.size > 1 else None,
        "peak_freq_hz": round(float(freqs[peak_f]), 3),
        "peak_time_s": float(times[peak_t]),
        "peak_power": float(power[peak_f, peak_t]),
    }


@server.tool()
def rope_pulse(
    duration_s: float,
    k_hz: float,
    J_hz: float,
    dt_s: float,
    out_path: str,
) -> dict[str, Any]:
    """Sample the analytic finite-time ROPE transfer (JMR 2003) and report its efficiency.

    ``duration_s`` is the allowed transfer time T, ``k_hz`` the I-spin
    relaxation rate and ``J_hz`` the scalar coupling, both in Hz (so
    ``n = k_hz / J_hz``), ``dt_s`` the sampling step. Writes a CSV with columns
    time_s,u1,u2,amplitude_hz,phase_rad (``u1``/``u2`` dimensionless controls,
    RF amplitude as nutation frequency in Hz, phase pi/2 = y, 0 = x). Returns
    ``efficiency`` (transfer efficiency at this duration; 1.0 lossless),
    ``g_unconstrained`` (the infinite-time ROPE bound sqrt(1+n^2)-n),
    ``t_crit_s`` (below it the pulse degenerates to INEPT), the switching time
    and the ROPE/INEPT gain.
    """
    validate_positive("duration_s", duration_s)
    validate_positive("dt_s", dt_s)
    n = rope_n(k_hz, J_hz)
    sampled = rope_waveform(duration_s, n, J_hz, dt_s)
    out = _output_file(out_path)
    amplitude_hz = sampled["amplitude"] / (2.0 * math.pi)
    lines = ["time_s,u1,u2,amplitude_hz,phase_rad"]
    for t, u1, u2, amp, ph in zip(
        sampled["times"], sampled["u1"], sampled["u2"], amplitude_hz, sampled["phase"], strict=True
    ):
        lines.append(f"{t:.12e},{u1:.9e},{u2:.9e},{amp:.9e},{ph:.9e}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    t_crit = rope_Tcrit(n, J_hz)
    return {
        "out_path": str(out),
        "n_steps": int(sampled["times"].size),
        "n": n,
        "efficiency": round(rope_finite_efficiency(duration_s, n, J_hz), 6),
        "g_unconstrained": round(rope_g(n), 6),
        "gain_over_inept": round(rope_gain_over_inept(n, J_hz), 6),
        "t_crit_s": t_crit,
        "switching_time_s": (
            rope_switching_time(duration_s, n, J_hz) if duration_s > t_crit and n > 0.0 else None
        ),
        "max_amplitude_hz": round(float(np.max(amplitude_hz)), 6),
    }


@server.tool()
def crop_pulse(
    ka_hz: float,
    kc_hz: float,
    J_hz: float,
    dt_s: float,
    out_path: str,
    truncation_window_s: float | None = None,
) -> dict[str, Any]:
    """Sample the analytic truncated CROP pulse (PNAS 2003) and report its efficiency.

    ``ka_hz``/``kc_hz`` are the DD-CSA cross-correlated relaxation rates for
    longitudinal/transverse terms and ``J_hz`` the coupling, all in the same
    Hz convention (``ka_hz >= kc_hz``); ``dt_s`` is the sampling step.
    ``truncation_window_s`` defaults to ``1/sqrt(ka^2-kc^2)`` and must be given
    when ``kc_hz == ka_hz`` (the untruncated pulse is infinitely long). Writes
    a CSV with columns time_s,amplitude_hz,irradiation_freq_hz (times centred
    on zero). Returns ``eta`` (Iz -> 2IzSz efficiency, 1.0 lossless),
    ``zeta``, the constant amplitude and irradiation offset in Hz, the window
    and the number of samples.
    """
    validate_positive("dt_s", dt_s)
    params = crop_pulse_params(ka_hz, kc_hz, J_hz)
    window = params.truncation_window if truncation_window_s is None else truncation_window_s
    if not math.isfinite(window):
        raise ValueError(
            "truncation_window_s is required when kc_hz == ka_hz (the untruncated CROP pulse "
            "is infinitely long)"
        )
    sampled = crop_waveform(ka_hz, kc_hz, J_hz, dt_s, window)
    out = _output_file(out_path)
    lines = ["time_s,amplitude_hz,irradiation_freq_hz"]
    for t, amp, freq in zip(
        sampled["times"], sampled["amplitude"], sampled["irrad_freq"], strict=True
    ):
        lines.append(f"{t:.12e},{amp:.9e},{freq:.9e}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "out_path": str(out),
        "n_steps": int(sampled["times"].size),
        "eta": round(crop_eta(ka_hz, kc_hz, J_hz), 6),
        "zeta": round(crop_zeta(ka_hz, kc_hz, J_hz), 6),
        "amplitude_hz": params.amplitude,
        "irradiation_freq_hz": params.irradiation_freq_hz,
        "truncation_window_s": window,
    }


def main() -> None:
    server.run()  # stdio


if __name__ == "__main__":
    main()
