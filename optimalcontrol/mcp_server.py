"""MCP server exposing the Seedless pulse-design workflow.

Run with ``optimalcontrol-mcp`` (or ``python -m optimalcontrol.mcp_server``);
speaks MCP over stdio. Requires the optional ``mcp`` dependency
(``pip install "optimalcontrol-nmr[mcp]"``).

Large arrays never cross the MCP boundary: ``design_seedless_pulse`` writes a
Bruker ``.shape`` file plus a design JSON (spec + phases) and returns paths and
summary numbers; ``bloch_offset_profile`` reloads that JSON to verify the pulse
with the independent Bloch model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from optimalcontrol import __version__
from optimalcontrol.bloch import propagate_bloch_ensemble
from optimalcontrol.ocseed import Band, SeedlessSpec, _axis_bloch

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
        "Design NMR RF pulses via the Seedless (band/restraint) route of the "
        "optimalcontrol package. Units: band edges and carrier in ppm, RF in Hz, "
        "duration in seconds. Workflow: design_seedless_pulse (optimise, verify, "
        "export), then bloch_offset_profile on the returned design_json to inspect "
        "the offset response."
    ),
)


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

    Runs ``n_seeds`` random starts and keeps the best (the cost is non-convex),
    verifies the winner on a dense grid with an independent Bloch model, and
    writes ``<out_dir>/<name>.shape`` (Bruker) plus ``<out_dir>/<name>.json``
    (spec + phases, input for bloch_offset_profile).

    In the returned ``worst_case`` dict, 1.0 is perfect — except ``xycite``
    entries, which report leftover |mz| where 0.0 is perfect. A negative value
    means a wrong-sign local minimum: retry with more n_seeds, or give the
    problem more duration_s / n_steps / rf_max_hz.
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

    worst_case = spec.evaluate(spec.waveform_xy(best_phases))

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
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
def bloch_offset_profile(
    design_json: str,
    offset_lo_hz: float,
    offset_hi_hz: float,
    n_points: int = 101,
    init: str = "z",
    b1_scale: float = 1.0,
) -> dict[str, Any]:
    """Propagate a designed pulse across offsets with the independent Bloch model.

    ``design_json`` is the file written by design_seedless_pulse. Offsets are Hz
    relative to the carrier. ``init`` is the starting Bloch axis ("x", "y", "z",
    or negated). Returns the final magnetisation per offset; regions not covered
    by any band are unconstrained by design.
    """
    design = json.loads(Path(design_json).expanduser().read_text())
    spec = _make_spec(design["spec"])
    waveform = spec.waveform_xy(np.asarray(design["phases_rad"], dtype=np.float64))
    offsets = np.linspace(offset_lo_hz, offset_hi_hz, n_points)
    final = propagate_bloch_ensemble(
        _axis_bloch(init),
        waveform,
        offsets,
        np.array([b1_scale]),
        spec.rf_max_hz,
        spec.dt,
    )[0]
    return {
        "offsets_hz": [round(float(x), 3) for x in offsets],
        "mx": [round(float(x), 4) for x in final[:, 0]],
        "my": [round(float(x), 4) for x in final[:, 1]],
        "mz": [round(float(x), 4) for x in final[:, 2]],
    }


def main() -> None:
    server.run()  # stdio


if __name__ == "__main__":
    main()
