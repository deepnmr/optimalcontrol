# Paper Figure Guide

Example scripts in `examples/` reproduce the numerical curves and pulse shapes
used by the ROPE and CROP paper guides. Each script has a `run()` function for
tests and saves a PNG under `examples/output/` when executed.

Run one figure script directly:

```bash
python examples/rope_sodium_formate.py
```

Or regenerate all committed example outputs:

```bash
for script in examples/*.py; do
  python "$script"
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
| `examples/crop_regression_params.py` | `examples/output/crop_regression_params.png` | Stored CROP regression parameter values used by `tests/test_crop.py` |

## Snapshot Tests

`tests/test_examples.py` imports every example module, calls `run()`, and checks
the returned numerical array against `examples/expected/*.npz`. When an example
is intentionally changed, regenerate the corresponding `.npz` snapshot by
running the example through its `run()` function and storing the returned array
under key `output`.

The scripts use `matplotlib.use("Agg")` so they can run in headless CI
environments.
