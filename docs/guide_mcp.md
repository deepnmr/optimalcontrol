# MCP Server Guide

`optimalcontrol-mcp` (module `optimalcontrol.mcp_server`) exposes the package to
LLM clients over the [Model Context Protocol](https://modelcontextprotocol.io/)
on stdio. It needs the optional `mcp` dependency
(`pip install "optimalcontrol-nmr[mcp]"`) and is launched on demand by the
client; nothing runs in the background. Register it with any MCP-capable
client, e.g. `claude mcp add optimalcontrol -- optimalcontrol-mcp`, or install
the bundled Claude Code plugin (see the README).

Design rules shared by every tool:

- **Units.** RF amplitudes and offsets are in Hz (offsets relative to the
  carrier: `offset_hz = (ppm - carrier_ppm) * spectrometer_mhz`); Seedless band
  edges and the carrier are in ppm; durations and time steps are in seconds;
  phases are reported in degrees; waveform amplitudes are fractions of
  `rf_max_hz` (1.0 = the RF cap).
- **Files, not arrays.** Tools write waveforms, trajectories and spectrograms to
  files and return paths plus summary numbers. Design tools write a Bruker
  shape-library `.shape` (amplitude in percent of `rf_max_hz`, phase in degrees,
  provenance tags `##$..._DURATION_S` / `..._RF_HZ`) and a *design JSON* that
  the other tools accept as input.
- **Independent verification.** Every fidelity quoted next to a design comes
  from `optimalcontrol.bloch`, a Bloch-vector model separate from the GRAPE /
  Seedless engines that produced the pulse.
- **Validation.** Inputs are checked Python-side with stable error strings
  (`rf_max_hz must be positive`, `state label must be one of Ix, Iy, Iz or x,
  y, z (optionally signed)`, ...); nothing is written when a call is rejected.

## Pulse files the tools read

`inspect_waveform`, `convert_waveform`, `bloch_offset_profile`,
`bloch_trajectory` and `waveform_spectrogram` all accept:

| File | Detected by | Duration / RF known? |
|---|---|---|
| Seedless design JSON (`design_seedless_pulse`) | `phases_rad` key | yes |
| GRAPE design JSON (`design_grape_pulse`) | `waveform_xy` key | yes |
| Waveform JSON (`optimalcontrol.io.export_json`, `convert_waveform ... json`) | `channels`/`data` keys, two channels x/y | duration from the time axis; RF only if `metadata.rf_max_hz` |
| Waveform CSV (`io.export_csv`, `convert_waveform ... csv`) | `.csv` suffix, header `time,x,y` | duration from the time axis; RF unknown |
| Bruker shape-library file | any other suffix (`.shape`, `.dx`, ...) | from `##$OCSEED_*` / `##$OPTIMALCONTROL_*` tags when written by this package, otherwise unknown |

Whenever a file lacks the duration or RF calibration, pass `duration_s` (s) and
`rf_max_hz` (Hz) explicitly; those arguments also override values found in the
file.

## Tools

### `design_seedless_pulse`

Constant-amplitude phase-only design via the Seedless route
(`optimalcontrol.ocseed`).

| Argument | Unit / type | Meaning |
|---|---|---|
| `spectrometer_mhz` | MHz | 1H frequency used to turn ppm into Hz |
| `carrier_ppm` | ppm | transmitter position |
| `rf_max_hz` | Hz | constant RF amplitude (nutation frequency) |
| `duration_s`, `n_steps` | s, int | pulse length and number of steps |
| `bands` | list of dicts | `ppm_lo`, `ppm_hi`, `restraint` (`s2s`, `universal`, `xycite`, `suppress`) plus restraint fields (`init`/`targ` axis labels, `rotation` `[axis, deg]`, `per_step`), optional `n_offsets`, `weight` |
| `out_dir`, `name` | path | writes `<name>.shape` and `<name>.json` |
| `n_seeds`, `max_iter` | int | random restarts kept best; iterations per restart (universal rotations about x or y start from a mirrored half-length pulse, see user manual §4-3) |
| `b1_scales`, `b1_weights` | lists | B1 ensemble (default 0.95/1.00/1.03 with 0.25/0.5/0.25) |

Returns `shape_file`, `design_json`, `infidelity`, `worst_case` per band over
the dense verification grid and all B1 scales (1.0 perfect; `xycite` reports
max leftover |mz|, 0.0 perfect), `dt_s`, `warnings`.

### `design_grape_pulse`

Full amplitude+phase GRAPE (`optimalcontrol.grape` / `optimizers.run_grape`)
for an isolated spin-1/2 over an offset × B1 ensemble.

| Argument | Unit / type | Meaning |
|---|---|---|
| `init`, `targ` | lists of labels | `Ix`, `Iy`, `Iz` with optional sign, paired index by index. One pair = state-to-state (`["Iz"]` → `["-Iz"]`); three pairs = universal rotation (180x: `["Ix","Iy","Iz"]` → `["Ix","-Iy","-Iz"]`). The mean real fidelity over pairs is maximised. |
| `rf_max_hz` | Hz | RF cap; amplitude above it is penalised (SNSA, weight `amplitude_penalty_weight`, default 0.05) and hard-clipped before verification/export |
| `duration_s`, `n_steps` | s, int | pulse length and steps |
| `offset_lo_hz`, `offset_hi_hz`, `n_offsets` | Hz, int | design offsets (`n_offsets=1` uses the centre) |
| `b1_scales` | list | RF multipliers, e.g. `[0.9, 1.0, 1.1]`; 1 or ≥3 entries (the engine reads exactly two as per-channel powers) |
| `method`, `seed`, `max_iter` | str, int, int | `lbfgs` or `newton`; seed of the random starting waveform (three pairs forming a rotation about x or y start from a mirrored half-length pulse); iteration cap |
| `out_dir`, `name` | path | writes `<name>.shape` (variable amplitude) and `<name>.json` |

Returns `fidelity` (GRAPE ensemble mean on the design grid after clipping),
`bloch_mean` (same grid, Bloch model — agrees to ~1e-6), `bloch_worst` with
`bloch_worst_offset_hz` / `bloch_worst_b1_scale` (minimum over a 201-point
offset grid × all B1 scales), `n_iter`, `n_feval`, `converged`, `reason`,
`max_amplitude_before_clip`, `clipped_steps`, `dt_s`, `warnings`.

Two-spin problems (`2IzSz` labels, J coupling, relaxation, two RF channels)
are not exposed through MCP: they have no single Bloch verification model and
need the full `ControlProblem` API (`docs/guide_grape.md`,
`docs/guide_spin_system.md`).

### `bloch_offset_profile`

Final magnetisation versus offset for any design JSON, with the Bloch model.

| Argument | Unit / type | Meaning |
|---|---|---|
| `design_json` | path | Seedless or GRAPE design JSON |
| `offset_lo_hz`, `offset_hi_hz`, `n_points` | Hz, int | evaluation grid |
| `init` | label | starting state: `x`/`y`/`z` or `Ix`/`Iy`/`Iz`, optionally negated |
| `b1_scale` | float | RF multiplier for the returned `mx`/`my`/`mz` |
| `targ` | label, optional | adds `transfer` (projection on the target, 1.0 perfect) and `summary` (`worst_transfer`, `worst_offset_hz`, `worst_b1_scale`, `mean_transfer`) |
| `b1_scales` | list, optional | requires `targ`; extends `summary` to the worst case over offsets × these scales |

Without `targ` the output is exactly the pre-0.6 contract: `offsets_hz`,
`mx`, `my`, `mz`.

### `inspect_waveform`

`path`, optional `duration_s`, `rf_max_hz`. Returns `format` (`seedless`,
`grape`, `json`, `csv`, `bruker`), `n_steps`, `duration_s`, `dt_s`,
`rf_max_hz`, amplitude statistics as fractions of `rf_max_hz`
(`max_amplitude`, `min_amplitude`, `mean_amplitude`, `constant_amplitude`,
`max_rf_hz`), and phase statistics in degrees (`phase_min_deg`,
`phase_max_deg`, `n_distinct_phases`, `binary_phase`).

### `convert_waveform`

`path`, `out_path`, `format` (`bruker`, `csv`, `json`), optional `duration_s`,
`rf_max_hz`, `title`. `csv`/`json` need a known duration; `bruker` writes
duration/RF tags when they are known. Returns the output path and the
resolved `n_steps`, `duration_s`, `rf_max_hz`.

### `bloch_trajectory`

`path`, `offsets_hz` (list, Hz), `out_path` (`.csv` or `.json`), optional
`init` (label, default `z`), `b1_scale`, `duration_s`, `rf_max_hz`. Writes the
Bloch vector after every step: CSV columns `step,time_s,offset_hz,mx,my,mz`
(one row per step per offset) or JSON with `time_s`, `offsets_hz` and
`mx`/`my`/`mz` shaped `[n_steps+1][n_offsets]`. Returns the path, sizes and
the final vector per offset.

### `waveform_spectrogram`

`path`, `out_path` (JSON), optional `duration_s`, `rf_max_hz`. Runs
`optimalcontrol.analysis.spectrogram_data` on the complex envelope `x + i y`
(scaled to Hz when the RF calibration is known). The file holds `times_s`,
`freqs_hz` (two-sided, relative to the carrier) and `power[n_freqs][n_times]`;
the return value gives grid sizes, `freq_resolution_hz` and the strongest bin.

### `rope_pulse`

`duration_s` (T), `k_hz` (I-spin relaxation rate), `J_hz`, `dt_s`, `out_path`.
`n = k_hz / J_hz`. Writes CSV columns `time_s,u1,u2,amplitude_hz,phase_rad`
(`amplitude_hz` is the RF nutation frequency; phase π/2 = y, 0 = x). Returns
`efficiency` at this duration, `g_unconstrained = sqrt(1+n²) − n`,
`gain_over_inept`, `t_crit_s`, `switching_time_s` (null when the pulse
degenerates to INEPT) and `max_amplitude_hz`. Theory in
`docs/guide_rope_crop.md`.

### `crop_pulse`

`ka_hz`, `kc_hz` (DD–CSA cross-correlated rates, `ka_hz >= kc_hz`), `J_hz`,
`dt_s`, `out_path`, optional `truncation_window_s` (default
`1/sqrt(ka²−kc²)`; required when `kc_hz == ka_hz`). Writes CSV columns
`time_s,amplitude_hz,irradiation_freq_hz` with times centred on zero. Returns
`eta` (Iz → 2IzSz efficiency), `zeta`, the constant `amplitude_hz`,
`irradiation_freq_hz` (= −J/2) and the window.

## Typical conversation

1. `design_grape_pulse(init=["Iz"], targ=["-Iz"], rf_max_hz=10000,
   duration_s=5e-4, n_steps=100, offset_lo_hz=-6000, offset_hi_hz=6000,
   n_offsets=25, b1_scales=[0.9, 1.0, 1.1], out_dir="~/pulses", name="inv")`
2. `bloch_offset_profile(design_json="~/pulses/inv.json", offset_lo_hz=-8000,
   offset_hi_hz=8000, n_points=161, init="Iz", targ="-Iz",
   b1_scales=[0.85, 1.0, 1.15])` to see how the profile degrades outside the
   design band and under larger B1 miscalibration.
3. `bloch_trajectory(path="~/pulses/inv.json", offsets_hz=[0, 3000],
   out_path="~/pulses/inv_traj.csv")` to plot the time course.
4. `convert_waveform(path="~/pulses/inv.json", out_path="~/pulses/inv.csv",
   format="csv")` for other software.

## Running the server by hand

```bash
optimalcontrol-mcp                      # stdio; speaks JSON-RPC on stdin/stdout
python -m optimalcontrol.mcp_server     # same
```

The tools are plain functions: `from optimalcontrol.mcp_server import
design_grape_pulse` lets you call them from Python or tests without any
transport (`tests/test_mcp_server.py` does exactly that).
