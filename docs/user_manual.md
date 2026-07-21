# optimalcontrol — User Manual

A step-by-step guide for someone new to both NMR pulse design and this package.
Follow it top to bottom and you will install the package, design a pulse by two
different routes, understand what came out, and write it to a file.

Every command and code block below was executed against this repository before
being written down. Where a number is a *physical* result it is quoted exactly;
where a number tracks the release (versions, test counts) this manual tells you
what to look for instead of pinning a value that will rot.

For theory and per-subsystem API detail, see the `guide_*.md` files listed in
[§12](#12-where-to-go-next).

---

## Contents

0. [What this package does, and which route you need](#0-what-this-package-does)
1. [Install and verify](#1-install-and-verify)
2. [The concepts that bite](#2-the-concepts-that-bite)
3. [Route A — your first GRAPE pulse](#3-route-a--your-first-grape-pulse)
4. [Route B — your first Seedless pulse](#4-route-b--your-first-seedless-pulse)
5. [Reading the results](#5-reading-the-results)
6. [Writing pulses to files](#6-writing-pulses-to-files)
7. [Making pulses robust](#7-making-pulses-robust)
8. [Analytical routes — ROPE and CROP](#8-analytical-routes--rope-and-crop)
9. [Running the bundled examples](#9-running-the-bundled-examples)
10. [Units cheat sheet](#10-units-cheat-sheet)
11. [Troubleshooting](#11-troubleshooting)
12. [Where to go next](#12-where-to-go-next)

---

## 0. What this package does

`optimalcontrol` designs the **RF pulse waveforms** applied to NMR spins. You
state what you want — "take this spin state to that one, across this range of
chemical shifts, without exceeding this RF power" — and the package produces the
time-dependent waveform that does it.

There are four routes in. Pick by what you are trying to do:

| Route | Nature | Use it when |
|---|---|---|
| **GRAPE** | Numerical optimisation | You want arbitrary targets and full control over the problem. **Start here.** |
| **Seedless** (`ocseed`) | Declarative bands → phase-only optimisation | You have an isolated spin-1/2 and think in terms of chemical-shift bands (e.g. invert methyls, hold water) |
| **ROPE** | Analytical formulas | Heteronuclear transfer under relaxation |
| **CROP** | Analytical formulas | Cross-correlated relaxation-optimised pulses |

GRAPE and Seedless optimise; ROPE and CROP evaluate closed-form expressions and
hand you a waveform directly.

The heavy numerics (propagation, gradients, Bloch profiles) run in a parallel
Rust extension. If Rust is unavailable the package falls back to NumPy/SciPy
automatically and gives the same answers, more slowly. You only ever write
Python.

---

## 1. Install and verify

### 1-1. Check the prerequisites

```bash
python3 --version
```

You need **Python 3.10 or newer**.

If no prebuilt wheel matches your environment, `pip` compiles the Rust extension
from source, which needs a Rust toolchain. On macOS with Homebrew:

```bash
brew install rust
```

`rustc --version` printing a version means you are ready. Runtime dependencies
are only `numpy` and `scipy`.

### 1-2. Install

Regular use, from PyPI:

```bash
pip install optimalcontrol
```

Working from a clone of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # adds pytest, ruff, mypy, matplotlib
```

### 1-3. Verify

```bash
python3 -c "import optimalcontrol; print(optimalcontrol.__version__)"
```

This prints a version string. **Any version from 0.4 on has the Seedless route
of [§4](#4-route-b--your-first-seedless-pulse);** older ones do not. Confirm with:

```bash
python3 -c "from optimalcontrol import Band, SeedlessSpec; print('seedless ok')"
```

Then run the test suite:

```bash
python3 -m pytest -q
```

**What to look for:** a summary line ending in `passed`, with no `failed` and no
`error`. The count itself grows as the suite grows — don't match it against a
number written in a document.

A faster end-to-end check that the numerics and the native kernel agree:

```bash
python3 -c "from optimalcontrol.ocseed import demo; demo()"
```

```
demo ok (grad rel 3.11e-09, infidelity 0.0022, worst 0.9984)
```

`grad rel` is the analytic gradient checked against finite differences. It
should be around `1e-9`; a large value means something is wrong with your build.

---

## 2. The concepts that bite

Five things account for most of the time beginners lose.

**1. Waveform orientation flips on export.** The waveform you *pass in* to the
optimiser is `(n_steps, n_channels)` — **time down the rows**. The `Waveform`
you *get back* holds `data` as `(n_channels, n_steps)` — **time along the
columns**. Both are correct; they are different objects. If an error message
mentions a shape, check which of the two you are holding.

**2. Generators must be anti-Hermitian.** On the GRAPE `dense` path you pass
operators as `-1j * Ix()`, not `Ix()`. Dropping the `-1j` is the single most
common mistake and it fails quietly — the optimiser stalls instead of raising.

**3. States get normalised.** Use `normalise_hs(...)` (Hilbert–Schmidt) on
density matrices before handing them to a `ControlProblem`.

**4. `pwr_levels` and `operators` must be the same length.** One RF level per
control channel. Mismatches raise immediately.

**5. `fidelity_mode` decides what "success" means.** `"real"` requires the sign
to match; `"abs2"` ignores global phase; `"imag"` takes the imaginary part.

---

## 3. Route A — your first GRAPE pulse

Goal: move one spin from `Iz` to `Ix` — a 90° rotation.

Save this as `my_first_pulse.py`:

```python
import numpy as np

import optimalcontrol
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.optimizers import run_grape
from optimalcontrol.states import normalise_hs

# (1) Fix the random seed so the run is reproducible
optimalcontrol.set_random_seed(0)

# (2) Describe the problem
cp = ControlProblem(
    drifts=[np.zeros((2, 2), dtype=np.complex128)],   # no free precession (on-resonance)
    operators=[np.complex128(-1j) * Ix(),             # x control channel
               np.complex128(-1j) * Iy()],            # y control channel
    rho_init=[normalise_hs(Iz())],                    # start:  Iz
    rho_targ=[normalise_hs(Ix())],                    # target: Ix
    pulse_dt=0.1,                                     # duration of one time slice
    pwr_levels=[1.0, 1.0],                            # RF strength per channel
    freeze=None,
    fidelity_mode="real",
    basis="dense",
)

# (3) Initial guess: 8 time steps, 2 channels, all zeros
wfm0 = np.zeros((8, 2), dtype=np.float64)

# (4) Optimise
waveform, result = run_grape(cp, wfm0, method="lbfgs", max_iter=100)

# (5) Inspect
print("converged?          :", result.converged)
print("final fidelity      :", round(result.fidelity_final, 6))
print("iterations          :", result.n_iter)
print("waveform channels   :", waveform.channels)
print("waveform data shape :", waveform.data.shape)
```

Run it:

```bash
python3 my_first_pulse.py
```

```
converged?          : True
final fidelity      : 1.0
iterations          : 5
waveform channels   : ['x', 'y']
waveform data shape : (2, 8)
```

**A fidelity at or near 1.0 means you succeeded.** You have designed an Iz → Ix
transfer pulse.

Note the shape flip from [§2](#2-the-concepts-that-bite) in action: you passed
`(8, 2)` and got back `data` of `(2, 8)`.

This problem is deliberately dimensionless — `pulse_dt=0.1` and
`pwr_levels=[1.0, 1.0]` are pedagogical, not seconds and hertz. Real units enter
in [§4](#4-route-b--your-first-seedless-pulse) and [§10](#10-units-cheat-sheet).

---

## 4. Route B — your first Seedless pulse

The Seedless route implements the formalism of Buchanan et al., *Nature
Communications* **16**, 7276 (2025). It is the right tool when your problem is
naturally stated as *bands of chemical shift, each wanting something different*.

Instead of assembling operators and states yourself, you declare bands:

- **`SeedlessSpec`** holds the hardware and timing: spectrometer frequency,
  carrier position, RF ceiling, pulse duration, number of steps.
- **`Band`** covers a ppm range and carries one **restraint** — what the spins in
  that band should do.

The optimisation variable is a single **constant-amplitude, phase-only**
waveform: the RF stays at `rf_max_hz` throughout and only the phase moves. B1
inhomogeneity is handled by a weighted average over several B1 scalings.

### 4-1. The four restraints

| Restraint | Meaning | Requires |
|---|---|---|
| `"universal"` | A full Bloch-sphere rotation, enforced as three cardinal transfers (X→UX, Y→UY, Z→UZ) | `rotation=(axis, angle_deg)` |
| `"s2s"` | One state-to-state transfer, e.g. `-y → y` | `init` and `targ` axis labels |
| `"xycite"` | Drive `Iz` into the transverse plane, don't care where it lands | — |
| `"suppress"` | Hold `Iz` on `Iz` (a water hold) | — |

`"suppress"` defaults to an end-of-pulse hold. Set `per_step=True` for the
paper's per-step `n²/2` form, which holds `Iz` after *every* prefix of the pulse
and therefore bounds transverse build-up during the pulse, not just at its end.

### 4-2. A single-band inversion

Start small: invert a ±8 ppm band, `-y → y`, in 120 µs at 600 MHz.

```python
from optimalcontrol import Band, SeedlessSpec

spec = SeedlessSpec(
    spectrometer_mhz=600.0,
    carrier_ppm=0.0,
    rf_max_hz=10_000.0,
    duration_s=120e-6,
    n_steps=40,
    bands=[Band(-8.0, 8.0, "s2s", n_offsets=9, init="-y", targ="y")],
)

phases, infidelity = spec.optimize(max_iter=150, seed=3)

print("dt (s)      :", spec.dt)
print("phases shape:", phases.shape)
print("infidelity  :", round(infidelity, 4))
print("worst-case  :", {k: round(v, 4) for k, v in
                        spec.evaluate(spec.waveform_xy(phases)).items()})
```

```
dt (s)      : 3e-06
phases shape: (40,)
infidelity  : 0.0022
worst-case  : {'band0:s2s': 0.9984}
```

Two different numbers, and the distinction matters:

- **`infidelity`** is the optimiser's own cost, averaged over the offsets and B1
  scalings you declared. Lower is better.
- **`evaluate()`** re-checks the waveform on a **dense offset grid** using an
  **independent Bloch forward model** — not the optimiser's math. Its keys are
  `"band<index>:<restraint>"` and it reports the **worst case** across the band.
  This is the number you trust.

That independence is the point: `evaluate()` can catch a waveform that scored
well on the coarse optimisation grid and fails between the grid points.

> **`"xycite"` reports the opposite polarity.** For every other restraint,
> `evaluate()` returns a fidelity where **1.0 is perfect**. For `"xycite"` it
> returns `max|mz|` — the leftover longitudinal component — where **0.0 is
> perfect**. A "bad-looking" 0.013 is an excellent `xycite` result.

### 4-3. Two bands: invert the methyls, keep the water

The real use of this API is bands that want different things at once.

```python
from optimalcontrol import Band, SeedlessSpec

spec = SeedlessSpec(
    spectrometer_mhz=600.0,
    carrier_ppm=4.7,               # carrier sits on water
    rf_max_hz=10_000.0,
    duration_s=3.0e-3,
    n_steps=200,
    bands=[
        Band(-3.0, 3.0, "universal", n_offsets=15, rotation=("x", 180.0)),
        Band(4.6, 4.8, "suppress", n_offsets=5),
    ],
)

# Phase-only optimisation from a random start is NOT globally convergent.
# Run several seeds and keep the best -- see the note below.
best_phases, best_infidelity = None, float("inf")
for seed in range(4):
    phases, infidelity = spec.optimize(max_iter=600, seed=seed)
    if infidelity < best_infidelity:
        best_phases, best_infidelity = phases, infidelity

print("best infidelity:", round(best_infidelity, 5))
print("worst-case     :", {k: round(v, 4) for k, v in
                           spec.evaluate(spec.waveform_xy(best_phases)).items()})
```

```
best infidelity: 0.00093
worst-case     : {'band0:universal': 0.996, 'band1:suppress': 0.9995}
```

The methyl band gets a 180° rotation about x good to ~0.996 worst-case, while
water `Iz` is held to ~0.9995 — all from one 3 ms phase-only waveform.

> **Why the multi-start loop is not optional.** On this exact problem, four of
> six random seeds converge to a **wrong-sign local minimum**: `evaluate()`
> returns a `universal` value near **−1.0**, meaning the band is being rotated
> to the *opposite* of the target. The cost function is non-convex and L-BFGS-B
> finds the nearest minimum, not the best one. A **negative** universal fidelity
> is the signature — see [§11](#11-troubleshooting). Always run several seeds,
> always check `evaluate()`, and never ship the first waveform that returns.

### 4-4. Calibrate B1 to your own hardware

`SeedlessSpec` defaults to `b1_scales=(0.95, 1.0, 1.03)` with
`b1_weights=(0.25, 0.5, 0.25)`, following the paper's Supplementary Note 2.8.
The main-text figures use a different spread (0.93/1.00/1.05).

**Both are properties of the authors' probe, not of physics.** Your probe's B1
distribution is yours. Measure it by nutation and pass your own:

```python
spec = SeedlessSpec(
    ...,
    b1_scales=(0.93, 1.0, 1.05),
    b1_weights=(0.25, 0.5, 0.25),   # must sum to 1.0
)
```

`b1_weights` must sum to 1 and match `b1_scales` in length; the constructor
rejects anything else.

### 4-5. Cross-checking the fast kernel

`fast=True` (the default) evaluates the objective and its exact gradient with an
analytic scaled-unitary spin-1/2 kernel. `fast=False` routes the same problem
through the general 4×4 Liouville GRAPE engine.

They agree to machine precision — that equivalence is what
`tests/test_ocseed.py::test_fast_kernel_matches_engine` pins. Use `fast=False`
when you suspect the kernel, not as a normal working mode; it is several times
slower.

---

## 5. Reading the results

`run_grape` returns `(waveform, result)`.

`result` is an `OptimResult`:

| Field | Meaning |
|---|---|
| `fidelity_final` | Final goal attainment; closer to 1.0 is better |
| `converged` | Whether a convergence criterion was met |
| `n_iter` | Iterations taken |
| `n_feval` | Objective evaluations |
| `reason` | Why the optimiser stopped, e.g. `grad_tol` |
| `history` | Fidelity per iteration, e.g. `[0.0, 0.5972, 0.9957, 0.9999, 1.0, 1.0]` |
| `wfm_final` | Final waveform, `(n_steps, n_channels)` |

`waveform` is a `Waveform`:

| Field | Meaning |
|---|---|
| `channels` | Channel names, e.g. `['x', 'y']` |
| `data` | Real array, `(n_channels, n_steps)` |
| `times` | Time stamp of each slice, `(n_steps,)` |
| `units` | Unit string, e.g. `a.u.` |
| `metadata` | The optimisation record: `basis`, `converged`, `fidelity_final`, `fidelity_mode`, `history`, `n_feval`, `n_iter`, `pulse_dt`, `pwr_levels`, `reason`, `source_wfm_shape` |
| `problem_hash` | Hash of the generating problem, for provenance |

`converged=True` and a high `fidelity_final` are different claims from *"this
pulse works"*. `converged` only says the optimiser stopped cleanly, and
`fidelity_final` is scored on the offsets and B1 values **you declared**. A pulse
that is perfect on a 9-point grid can fail between the points. Verify on a dense
grid with an independent model — `SeedlessSpec.evaluate()` in
[§4](#4-route-b--your-first-seedless-pulse), or `propagate_bloch_ensemble` in
[§7](#7-making-pulses-robust).

---

## 6. Writing pulses to files

### 6-1. Round-tripping a `Waveform`

```python
from optimalcontrol.io import export_csv, export_json, export_bruker, import_json

export_csv(waveform, "pulse.csv")        # human-readable table
export_json(waveform, "pulse.json")      # full metadata, lossless
export_bruker(waveform, "pulse.bruker")  # amplitude/phase(deg) interop stub

restored = import_json("pulse.json")     # round-trips data and problem_hash
```

- **CSV** — `time,x,y` columns. For eyeballing and plotting.
- **JSON** — channels, units, metadata and `problem_hash`. This is the format to
  archive; `import_json` restores it exactly.
- **Bruker** — see the warning below.

### 6-2. Read this before you put a shape on a spectrometer

There are **two** Bruker writers in this package and they are not
interchangeable.

> **`export_bruker()` is an interoperability stub, not a spectrometer-ready
> exporter.** Its own docstring, and a `##$OPTIMALCONTROL_LIMITATIONS` tag it
> writes into every file it produces, say so: it performs **no spectrometer
> calibration, no power normalisation, no vendor shape-parameter validation, and
> none of the safety checks required before loading a pulse on hardware.** It
> exists so a `Waveform` can round-trip through an amplitude/phase table. Do not
> treat its output as a shape file to load and run.

For an actual Bruker shape-library file, use the writer the bundled examples use
— `optimalcontrol.io.export_bruker_shape`, or, on the Seedless route, the
`export_shape` method that wraps it:

```python
path = spec.export_shape(best_phases, "methyl_180.shape", title="methyl 180 over water")
```

```
##TITLE= methyl 180 over water
##JCAMP-DX= 5.00 Bruker JCAMP library
##DATA TYPE= Shape Data
##ORIGIN= optimalcontrol
##OWNER= optimalcontrol
##MINX= 0.000000e+00
##MAXX= 1.000000e+02
##MINY= 0.000000e+00
##MAXY= 3.600000e+02
```

This writes amplitude in percent of your calibrated RF maximum and phase in
degrees, and tags the file with the duration, RF ceiling, spectrometer frequency
and carrier the pulse was designed for.

**Even then, no export path calibrates your probe or checks your hardware
limits.** The amplitude is a percentage of an RF maximum *you* are responsible
for having calibrated, and the pulse was designed against a B1 distribution you
supplied ([§4-4](#4-4-calibrate-b1-to-your-own-hardware)). Verify power levels
and duty cycle on your spectrometer before transmitting.

---

## 7. Making pulses robust

A pulse that only works exactly on resonance at exactly nominal power is not
useful. Real samples have a spread of resonance offsets, and real probes have
B1 inhomogeneity across the sample volume.

On the **Seedless** route this is built in: `Band(ppm_lo, ppm_hi, ...,
n_offsets=N)` spreads N spins across the band, and `b1_scales`/`b1_weights`
average over RF scalings ([§4-4](#4-4-calibrate-b1-to-your-own-hardware)).

On the **GRAPE** route you add ensemble axes to the `ControlProblem`:

- **Offsets** — set `offsets` (a list in Hz) and `offset_operators` (the operator
  for each axis).
- **B1** — give `pwr_levels` several strengths to create an RF ensemble axis.
- **Phase cycles** — `phase_cycle` averages over a phase cycle.
- **Penalties** — `penalties` constrains things like peak amplitude.

Multiple drifts, offsets and B1 values expand into a Cartesian-product ensemble,
and the optimiser maximises the average fidelity over every combination. See
`docs/guide_ensembles_penalties.md`.

### Checking an offset profile

`propagate_bloch_ensemble` is the independent Bloch forward model. Use it to
plot what your pulse actually does across offset:

```python
import numpy as np
from optimalcontrol import propagate_bloch_ensemble

offsets = np.linspace(-2000.0, 2000.0, 9)   # Hz from the carrier

final = propagate_bloch_ensemble(
    np.array([0.0, 0.0, 1.0]),         # start at Iz
    spec.waveform_xy(best_phases),     # (n_steps, 2), fractions of rf_max
    offsets,                           # Hz
    np.array([1.0]),                   # B1 scales
    spec.rf_max_hz,
    spec.dt,
)

print(final.shape)                # (n_b1, n_offsets, 3)
print(np.round(final[0, :, 2], 3))   # mz vs offset
```

```
(1, 9, 3)
[-0.999 -0.999 -0.994  0.015  1.     0.016  0.421  0.408  0.814]
```

Read that profile against the spec from [§4-3](#4-3-two-bands-invert-the-methyls-keep-the-water):
the carrier is at water (4.7 ppm), so offset `0` is water — `mz = 1.0`, held, as
the `suppress` band demanded. The methyl band (−3 to 3 ppm, i.e. −4620 to −1020
Hz from this carrier) sits at the negative end — `mz ≈ −0.999`, cleanly
inverted.

The junk at the positive end (`0.42`, `0.41`, `0.81`) is not a bug. **No band
was declared there, so nothing constrained it.** The optimiser optimises what
you asked for and nothing else. If you care about a region, give it a band.

---

## 8. Analytical routes — ROPE and CROP

These need no optimisation: the waveform comes from closed-form expressions.

**ROPE** — finite-time controls and RF waveform sampling:

```python
from optimalcontrol.rope import rope_g, rope_waveform

print(rope_g(2.0))                  # ROPE gain factor -> 0.2360679774997898

wf = rope_waveform(T=5e-3, n=2.0, J_hz=140.0, dt=1e-4)
print(sorted(wf.keys()))            # ['amplitude', 'phase', 'times', 'u1', 'u2']
print(len(wf["times"]))             # 50
```

The return is a dict: `times` (seconds), `u1`/`u2` (dimensionless controls),
`amplitude` (rad/s), `phase` (radians).

**CROP** — symmetrically truncated pulse waveform:

```python
from optimalcontrol.crop import crop_waveform

wf = crop_waveform(ka=0.0, kc=0.0, J_hz=140.0, dt=1e-4, truncation_window=5e-3)
print(sorted(wf.keys()))            # ['amplitude', 'irrad_freq', 'times']
print(len(wf["times"]))             # 50
```

Theory in `docs/guide_rope_crop.md`; the underlying papers in
`docs/source_notes.md`.

---

## 9. Running the bundled examples

`examples/` holds 24 paper-reproduction scripts. **Running one before you write
your own code is the fastest way to see a whole workflow.** Each is directly
executable as a module:

```bash
ls examples/*.py

python3 -m examples.reburp_pulse                  # textbook REBURP 180 reference plots
python3 -m examples.grape_broadband_180           # broadband 180 via GRAPE (cached, instant)
python3 -m examples.methyl_water_binary_symmetric_180   # the flagship methyl-over-water pulse
```

Outputs — plots and `.shape` files — land in `examples/output/`.

### The `--optimize` flag

Examples that involve a search ship a **cached** answer so they run instantly.
Pass `--optimize` to redo the search yourself (slow):

```bash
python3 -m examples.grape_broadband_180 --optimize
```

### The Pareto pair

Two mirror-image examples map the trade-off between RF power and pulse length
for a band-selective methyl 180 over water. There is no single best answer, only
a frontier:

| Peak RF | Minimum duration | |
|---|---|---|
| 10.0 kHz | 1.80 ms | `examples.methyl_water_reburp_minlength_180` |
| 9.0 kHz | 1.90 ms | |
| 7.5 kHz | 1.90 ms | |
| 7.0 kHz | 2.10 ms | |
| 6.0 kHz | 2.60 ms | `examples.methyl_water_reburp_minpower_180` |

The curve is flat from 10 kHz down to about 7.5 kHz, then duration climbs
steeply — so if your probe can take 7.5 kHz you get the short pulse nearly for
free, and below that you pay in length. Each example caches one end of the
frontier; `--optimize` re-runs the `(max-power, duration)` search that produced
it.

These are numerical grid-search results on a tested grid, not proofs of a
continuous global optimum.

---

## 10. Units cheat sheet

Units are where beginners lose the most time.

| Quantity | Unit |
|---|---|
| Public coupling constants (J), chemical shifts | **Hz** |
| `Band` band edges (`ppm_lo`, `ppm_hi`), `carrier_ppm` | **ppm** |
| `SeedlessSpec.rf_max_hz` | **Hz** |
| `SeedlessSpec.duration_s`, `dt` | **seconds** |
| `SeedlessSpec.optimize()` phases | **radians** |
| `SeedlessSpec.waveform_xy()` values | **fractions of `rf_max_hz`** |
| `export_shape` amplitude / phase in the file | **percent of calibrated RF max** / **degrees** |
| Relaxation fields on `RelaxationRates` | **rad/s** |
| `rope_waveform` `amplitude` / `phase` | **rad/s** / **radians** |
| `pulse_dt`, `pwr_levels` in the [§3](#3-route-a--your-first-grape-pulse) example | **dimensionless** (pedagogical only) |

To convert Hz to angular frequency, multiply by `2π` — e.g. `2 * np.pi * 140.0`
for a 140 Hz coupling. Dropping the `2π`, or the `-1j` from
[§2](#2-the-concepts-that-bite), is the leading cause of results that are wrong
rather than merely poor.

---

## 11. Troubleshooting

**Fidelity stalls low, immediately, with a tiny `n_iter`.**
Check the generators are anti-Hermitian (`-1j * Ix()`, not `Ix()`). Check states
are `normalise_hs(...)`-normalised. Then check the pulse can physically reach the
target at all — a transfer needs enough time and enough RF; try raising
`pulse_dt`, the step count, or `pwr_levels`.

**`evaluate()` returns a *negative* `universal` value (near −1.0).**
The optimiser converged to a wrong-sign local minimum: your band is being rotated
to the opposite of the target. This is expected behaviour from a bad random
start, not a bug — the cost is non-convex. Re-run with different `seed` values
and keep the best, as in [§4-3](#4-3-two-bands-invert-the-methyls-keep-the-water).
If every seed lands negative, the problem is likely infeasible as posed: give it
more `duration_s`, more `n_steps`, or a higher `rf_max_hz`.

**`optimize()` returns a good infidelity but `evaluate()` disagrees.**
Trust `evaluate()`. The optimiser scored your coarse `n_offsets` grid; `evaluate`
re-checks on a dense grid with an independent Bloch model. Raise `n_offsets` and
re-optimise.

**A `"xycite"` band looks terrible at ~0.01.**
It isn't. `xycite` reports `max|mz|`, where **0.0 is perfect**. See
[§4-2](#4-2-a-single-band-inversion).

**`pwr_levels length ... must match operator count`.**
One entry in `pwr_levels` per entry in `operators`.

**`waveform must have shape (n_steps, n_channels)`.**
Optimiser input is time-down-the-rows. You are probably holding an exported
`waveform.data`, which is transposed. See [§2](#2-the-concepts-that-bite).

**`b1_weights must sum to 1`.**
Exactly what it says; `b1_scales` and `b1_weights` must also be the same length.

**Comparing against the pure-NumPy path (no Rust).**

```bash
OPTIMALCONTROL_DISABLE_RUST=1 python3 my_first_pulse.py
```

Both paths give the same answers to well within physical significance, but not
bit-for-bit — expect agreement in the first several digits, not the last. Rust is
roughly an order of magnitude faster on realistic problems (the [§4-3](#4-3-two-bands-invert-the-methyls-keep-the-water)
optimisation takes ~0.8 s native versus ~10 s in fallback on one development
machine). If a result *only* reproduces on one path, that is a bug worth
reporting.

**Lint and type checks.**

```bash
ruff check .
mypy optimalcontrol
```

**When stuck**, the working code in `examples/` is the most reliable usage
reference in the repository — it is executed by the test suite, so it cannot rot
silently.

---

## 12. Where to go next

| Document | Covers |
|---|---|
| `docs/guide_spin_system.md` | Building spin systems |
| `docs/guide_states.md` | States and fidelity |
| `docs/guide_operators.md` | Operator utilities |
| `docs/guide_grape.md` | GRAPE in depth (Hilbert and Liouville paths) |
| `docs/guide_ensembles_penalties.md` | Ensembles and penalties |
| `docs/guide_rope_crop.md` | ROPE/CROP theory |
| `docs/guide_paper_figures.md` | Reproducing the paper figures |
| `docs/equations.md` | The underlying equations |
| `docs/source_notes.md` | The source papers and how this package maps to them |
| `docs/spinach_mapping.md` | Mapping to the Spinach (MATLAB) API |

You have now installed the package, optimised a pulse two different ways, read
the results, checked them against an independent model, and written a shape file.
That is the whole workflow — everything else is detail.
