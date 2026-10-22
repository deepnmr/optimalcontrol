# Paper Figure Guide

Example scripts in `examples/` reproduce the numerical curves and pulse shapes
used by the ROPE and CROP paper guides. Each script has a `run()` function for
tests and saves a PNG under `examples/output/` when run from the repository root with
`python -m examples.<name>`.
The PNGs are byte-stable across the Rust and NumPy backends for the recorded
rendering environment (matplotlib 3.11.2 with the Agg backend, used to regenerate
the committed figures), because Bloch-derived curves are quantised to 12 decimals
(`examples/_figures.quantise`) before plotting; a different matplotlib or FreeType
version can still change the bytes.

Run one figure script directly:

```bash
python -m examples.rope_sodium_formate
```

Or regenerate all committed example outputs:

```bash
for script in examples/[a-z]*.py; do
  python -m "examples.$(basename "$script" .py)"
done
```

## ROPE Examples

| Script | Output | What it reproduces |
|---|---|---|
| `examples/rope_efficiency_vs_n.py` | `examples/output/rope_efficiency_vs_n.png` | JMR ROPE `g(n)` versus INEPT maximum efficiency over `n = kI / J` |
| `examples/rope_gain_Ia_2IbSc.py` | `examples/output/rope_gain_Ia_2IbSc.png` | ROPE gain over INEPT for the `Ia -> 2IbSc` transfer |
| `examples/rope_inphase_gain.py` | `examples/output/rope_inphase_gain.png` | Equal-rate in-phase ROPE efficiency `g_in(n) = g(n)^2` |
| `examples/rope_finite_time_efficiency.py` | `examples/output/rope_finite_time_efficiency.png` | Finite-time ROPE efficiency `g_T` versus transfer duration with the INEPT/ROPE branch change at `Tcrit` |
| `examples/rope_finite_time_controls.py` | `examples/output/rope_finite_time_controls.png` | Three-phase finite-time ROPE controls `u1`, `u2`, and RF amplitude |
| `examples/rope_sodium_formate.py` | `examples/output/rope_sodium_formate.png` | Sodium-formate `13C-1H` worked example with `J = 193 Hz` and `T2 = 1.4 ms` |
| `examples/rope_hard_pulse.py` | `examples/output/rope_hard_pulse.png` | Hard-pulse approximation of the finite-time ROPE boundary arcs and flip-angle calculation |
| `examples/jmr2005_fig5_rope.py` | `examples/output/jmr2005_fig5_rope.png` | JMR 2005 GRAPE paper Fig. 5 conditions: `Iz -> 2IzSz`, `J = 194 Hz`, `k/J = 1`, `T = 0.408/J`, `N = 75` |

## Broadband GRAPE Pulse Examples

| Script | Output | What it reproduces |
|---|---|---|
| `examples/sciadv2023_fig1_ur180.py` | `examples/output/sciadv2023_fig1_ur180.png`, `examples/output/sciadv2023_fig1_ur180.shape` | Sci. Adv. 2023 Fig. 1-style low-power phase-modulated UR-180 pulse: `540 us`, `7.5 kHz`, `±6.3 kHz`, `±15% B1` |

## CROP Examples

| Script | Output | What it reproduces |
|---|---|---|
| `examples/crop_efficiency_vs_ka.py` | `examples/output/crop_efficiency_vs_ka.png` | PNAS CROP efficiency `eta` versus `ka/J` for several `kc/ka` ratios |
| `examples/crop_truncated_waveform.py` | `examples/output/crop_truncated_waveform.png` | Truncated CROP amplitude and irradiation-frequency waveforms for two parameter sets |
| `examples/crop_lossless_limit.py` | `examples/output/crop_lossless_limit.png` | Approach to the decoherence-free limit as `kc/ka -> 1` |
| `examples/crop_regression_params.py` | `examples/output/crop_regression_params.png` | Stored CROP regression parameter values |

## Pulse Design Examples

| Script | Output | What it does |
|---|---|---|
| `examples/grape_broadband_180.py` | `examples/output/grape_broadband_180.png`, `examples/output/grape_broadband_180.shape` | Generic 72-step, 540 us broadband UR-180 GRAPE pulse |
| `examples/phase_only_iy_inversion_10khz.py` | `examples/output/phase_only_iy_inversion_10khz.png`, `examples/output/phase_only_iy_inversion_10khz.shape` | Shortest phase-only 10 kHz `Iy -> -Iy` pulse found over 20 kHz |
| `examples/hmqc_oc_180_artifact.py` | `examples/output/hmqc_oc_180_artifact.png`, `examples/output/hmqc_oc_180_artifact.shape` | Band-selective OC 180 that suppresses methyl-HMQC artifacts (Kay 2019) |
| `examples/inept_hardI_broadbandS6k.py` | `examples/output/inept_hardI_broadbandS6k_band_profile.png`, `examples/output/inept_hardI_broadbandS6k.shape` | INEPT-like `Iz -> 2IzSz` with hard I pulses and a broadband S 180 |
| `examples/methyl_water_binary_symmetric_180.py` | `examples/output/methyl_water_binary_symmetric_180.png`, `examples/output/methyl_water_binary_symmetric_180_time_evolution.png`, `examples/output/methyl_water_binary_symmetric_180.shape` | Shortest grid-searched 0/180 symmetric methyl 180 preserving water |
| `examples/methyl_water_reburp_180.py` | `examples/output/methyl_water_reburp_180.png`, `examples/output/methyl_water_reburp_180_time_evolution.png`, `examples/output/methyl_water_reburp_180.shape` | REBURP-style methyl 180 preserving water |
| `examples/methyl_water_reburp_minpower_180.py` | `examples/output/methyl_water_reburp_minpower_180.png` | Minimum max-power REBURP-style methyl 180 over water |
| `examples/methyl_water_reburp_minlength_180.py` | `examples/output/methyl_water_reburp_minlength_180.png` | Minimum-length REBURP-style methyl 180 over water at 10 kHz |
| `examples/methyl_pc9_sideband_180.py` | `examples/output/methyl_pc9_sideband_180.png`, `examples/output/methyl_pc9_sideband_180_regime.png` | Methyl Kay sideband analysis of a PC9 180 pulse |
| `examples/reburp_pulse.py` | `examples/output/reburp_pulse.png`, `examples/output/reburp_pulse_wide_profile.png`, `examples/output/reburp_pulse_wide_components.png`, `examples/output/reburp_pulse_time_evolution.png` | Textbook REBURP (Geen & Freeman) reference plots |
| `examples/compare_3919_reburp_myprofile.py` | `examples/output/compare_3919_reburp_myprofile.png` | My refocusing profile: 3-9-19 WATERGATE vs min-length REBURP |

## Snapshot Tests

`tests/test_examples.py` imports the 20 snapshot-listed examples, calls `run()`, and checks
the returned numerical array against `examples/expected/*.npz`. When an example
is intentionally changed, regenerate the corresponding `.npz` snapshot by
running the example through its `run()` function and storing the returned array
under key `output`.

The scripts use `matplotlib.use("Agg")` so they can run in headless CI
environments.
