# optimalcontrol User Manual (Step-by-Step Beginner's Guide)

This guide is written so that someone new to both NMR and Python packaging can
follow it **top to bottom** and end up with their first optimal-control pulse,
exported in a spectrometer-ready format. Every step pairs "commands/code to
copy and run" with "what you should see if it worked".

> Every code snippet in this manual has been executed and verified. For deeper
> theory and API detail, see the other `guide_*.md` documents in `docs/`.

---

## Contents

0. [What this package does (one paragraph)](#0-what-this-package-does)
1. [Installation](#1-installation)
2. [The 6 concepts you must know](#2-the-6-concepts-you-must-know)
3. [Your first pulse in 5 minutes (GRAPE)](#3-your-first-pulse-in-5-minutes-grape)
4. [Reading the results](#4-reading-the-results)
5. [Exporting to files (CSV / JSON / Bruker)](#5-exporting-to-files)
6. [Running the bundled examples](#6-running-the-bundled-examples)
7. [Analytical pulses (ROPE / CROP) quick recipes](#7-analytical-pulses-rope--crop)
8. [Making robust pulses (offset / B1 ensembles)](#8-making-robust-pulses)
9. [Units cheat sheet (the most common mistakes)](#9-units-cheat-sheet)
10. [Troubleshooting](#10-troubleshooting)
11. [Where to go next](#11-where-to-go-next)

---

## 0. What this package does

`optimalcontrol` is a Python package for designing the **RF pulse waveforms**
applied to NMR spins. You tell it "I want to move this spin state
(`rho_init`) to that state (`rho_targ`)", and it finds the time-dependent RF
amplitude/phase waveform that performs the transformation best.

It offers three approaches.

| Method | Nature | When to use |
|--------|--------|-------------|
| **GRAPE** | Numerical optimisation | Flexible, arbitrary targets. Start here |
| **ROPE** | Analytical formulas | Heteronuclear transfer under relaxation |
| **CROP** | Analytical formulas | Cross-correlated relaxation-optimised pulses |

The heavy computation (propagation, gradients) runs in a parallel Rust
extension; if Rust is unavailable it automatically falls back to NumPy/SciPy.
You only ever write Python.

---

## 1. Installation

### 1-1. Check the prerequisites

Run this in a terminal to confirm you have Python 3.10 or newer:

```bash
python3 --version
```

You want to see `Python 3.10.x` or later.

If no prebuilt wheel exists for your environment, the Rust extension has to be
compiled from source, so you also need a Rust toolchain. On macOS with
Homebrew:

```bash
brew install rust
```

If `rustc --version` prints a version, you are ready.

### 1-2. Create a virtual environment and install

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # includes the development tools
```

Regular users can also install straight from PyPI:

```bash
pip install optimalcontrol
```

### 1-3. Verify the installation

If this one-liner prints a version string, the install succeeded:

```bash
python3 -c "import optimalcontrol; print(optimalcontrol.__version__)"
```

Expected output:

```
0.3.0
```

To be thorough, run the full test suite:

```bash
python3 -m pytest -q
```

A line like `203 passed` means your environment is fully healthy.

---

## 2. The 6 concepts you must know

Before writing code, you only need to understand six terms.

1. **`ControlProblem`** — the box that holds "what to solve": drift/control
   operators, initial and target states, time step, RF power levels, and so on.

2. **Waveform shape** — always a 2-D array of shape `(n_steps, n_channels)`.
   **Rows are time slices, columns are control channels** (usually x and y).
   When in doubt, remember only this sentence.

3. **Generators** — the operators that drive time evolution. On the GRAPE
   `dense` path you pass them in anti-Hermitian form, e.g. `-1j * Ix()`.
   Forgetting the `1j` is the single most common beginner mistake.

4. **States** — vectors or square density matrices. Normalise density
   matrices with `normalise_hs(...)` (Hilbert–Schmidt normalisation).

5. **`fidelity_mode`** — how goal attainment is measured: one of `"real"`,
   `"imag"`, `"abs2"`. Use `"abs2"` when global phase doesn't matter and
   `"real"` when sign must match too.

6. **`pwr_levels` and `pulse_dt`** — the RF strength per channel and the
   duration of one time slice. The length of `pwr_levels` must equal the
   number of entries in `operators`.

These six are enough to build your first pulse.

---

## 3. Your first pulse in 5 minutes (GRAPE)

Goal: find a pulse that moves one spin from the `Iz` state to the `Ix` state
(equivalent to a 90-degree rotation).

Create a file called `my_first_pulse.py` and paste this in verbatim:

```python
import numpy as np

import optimalcontrol
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.optimizers import run_grape
from optimalcontrol.states import normalise_hs

# (1) Fix the random seed for reproducibility
optimalcontrol.set_random_seed(0)

# (2) Put the problem in its box
cp = ControlProblem(
    drifts=[np.zeros((2, 2), dtype=np.complex128)],          # no free precession (on-resonance)
    operators=[np.complex128(-1j) * Ix(),                    # x control channel
               np.complex128(-1j) * Iy()],                   # y control channel
    rho_init=[normalise_hs(Iz())],                           # start: Iz
    rho_targ=[normalise_hs(Ix())],                           # target: Ix
    pulse_dt=0.1,                                            # slice duration (dimensionless example)
    pwr_levels=[1.0, 1.0],                                   # RF strength per channel
    freeze=None,
    fidelity_mode="real",
    basis="dense",
)

# (3) Initial guess waveform: all zeros (8 steps x 2 channels)
wfm0 = np.zeros((8, 2), dtype=np.float64)

# (4) Run the optimisation
waveform, result = run_grape(cp, wfm0, method="lbfgs", max_iter=100)

# (5) Look at the results
print("converged?      :", result.converged)
print("final fidelity  :", round(result.fidelity_final, 6))
print("iterations      :", result.n_iter)
print("waveform channels :", waveform.channels)
print("waveform data shape :", waveform.data.shape)
```

Run it:

```bash
python3 my_first_pulse.py
```

Expected output (values may differ very slightly between environments):

```
converged?      : True
final fidelity  : 1.0
iterations      : 5
waveform channels : ['x', 'y']
waveform data shape : (2, 8)
```

**If `final fidelity` is close to 1.0, you succeeded.** You have just found an
Iz → Ix transfer pulse.

> A confusing point: the input `wfm0` has shape
> `(n_steps, n_channels) = (8, 2)`, but the exported `waveform.data` has shape
> `(n_channels, n_steps) = (2, 8)`. **Optimisation input has time on the rows;
> the exported waveform has time on the columns.** See concept 2.

---

## 4. Reading the results

`run_grape` returns two things: `waveform` (an exportable waveform) and
`result` (the optimisation report).

Key fields of `result` (an `OptimResult`):

| Field | Meaning |
|-------|---------|
| `fidelity_final` | Final goal attainment (closer to 1.0 is better) |
| `converged` | Whether a convergence criterion was met |
| `n_iter` | Number of iterations |
| `n_feval` | Number of objective evaluations |
| `reason` | String explaining why the optimiser stopped |
| `history` | Per-iteration fidelity record (list) |
| `wfm_final` | Final waveform, `(n_steps, n_channels)` |

Key fields of `waveform` (a `Waveform`):

| Field | Meaning |
|-------|---------|
| `channels` | Channel names, e.g. `['x', 'y']` |
| `data` | Real array of shape `(n_channels, n_steps)` |
| `times` | Time stamp of each slice |
| `units` | Unit string |
| `metadata` | Extra information |
| `problem_hash` | Hash of the generating problem (for provenance) |

To inspect the convergence curve:

```python
print(result.history)      # [f0, f1, ...] fidelity climbing over iterations
```

---

## 5. Exporting to files

You can save the waveform in three formats:

```python
from optimalcontrol.io import export_csv, export_json, export_bruker

export_csv(waveform, "pulse.csv")        # human-readable table
export_json(waveform, "pulse.json")      # preserves metadata completely
export_bruker(waveform, "pulse.bruker")  # Bruker amplitude/phase(deg) format
```

- **CSV**: times and channel values as a table — for quick visual checks.
- **JSON**: stores channels, units, metadata, and the problem hash; restore
  later with `import_json`.
- **Bruker**: converts the x/y channels to amplitude and phase (in degrees)
  for direct loading on the spectrometer.

To load back:

```python
from optimalcontrol.io import import_json
same_waveform = import_json("pulse.json")
```

---

## 6. Running the bundled examples

The package ships paper-reproduction examples in the `examples/` folder.
**Running one example before writing your own waveform** gives you the whole
workflow at a glance.

The textbook REBURP 180-degree band-selective pulse reference plots:

```bash
python3 -m examples.reburp_pulse
```

A broadband 180-degree pulse optimised with full GRAPE (uses a cached
solution, runs instantly):

```bash
python3 -m examples.grape_broadband_180
```

To re-run the optimisation from scratch, add `--optimize`:

```bash
python3 -m examples.grape_broadband_180 --optimize
```

Outputs (plots, `.shape` files) are written to `examples/output/`.

To list the available examples:

```bash
ls examples/*.py
```

---

## 7. Analytical pulses (ROPE / CROP)

This path gives you waveforms directly from formulas, with no optimisation.

**ROPE** — finite-time controls and RF waveform sampling:

```python
from optimalcontrol.rope import rope_g, rope_waveform

print(rope_g(2.0))   # ROPE gain factor

wf = rope_waveform(T=5e-3, n=2.0, J_hz=140.0, dt=1e-4)
print(sorted(wf.keys()))     # ['amplitude', 'phase', 'times', 'u1', 'u2']
print(len(wf["times"]))      # number of time slices
```

The return value is a dict: `times` (seconds), `u1`/`u2` (dimensionless
controls), `amplitude` (rad/s), `phase` (radians).

**CROP** — symmetrically truncated pulse waveform:

```python
from optimalcontrol.crop import crop_waveform

wf = crop_waveform(ka=0.0, kc=0.0, J_hz=140.0, dt=1e-4, truncation_window=5e-3)
print(sorted(wf.keys()))     # ['amplitude', 'irrad_freq', 'times']
```

For the theoretical background see `docs/guide_rope_crop.md`.

---

## 8. Making robust pulses

Real spectrometers have resonance offsets and B1 (RF strength)
inhomogeneity. Add ensemble axes to the `ControlProblem` and the optimiser
finds pulses that work well across all conditions simultaneously.

- **Offset robustness**: set `offsets` (a list in Hz) and `offset_operators`
  (the operators for each axis).
- **B1 robustness**: give `pwr_levels` several strengths to create an RF
  ensemble axis.
- **Phase cycles**: use `phase_cycle` to average over a phase cycle.

Combining multiple drifts/offsets/B1 values expands into a Cartesian-product
ensemble and the average fidelity over all combinations is optimised. For
details, including penalties (amplitude limits etc.), see
`docs/guide_ensembles_penalties.md`.

To plot offset profiles and check performance, use Bloch ensemble
propagation:

```python
from optimalcontrol.bloch import propagate_bloch_ensemble
```

(For concrete arguments, follow how `examples/grape_broadband_180.py`
actually uses it.)

---

## 9. Units cheat sheet

Units are where beginners get stuck most often. The rules:

| Quantity | Unit |
|----------|------|
| Public coupling constants (J), chemical shifts | **Hz** |
| Relaxation fields on `RelaxationRates` | **rad/s** |
| `rope_waveform` `amplitude` | **rad/s** (angular amplitude) |
| `rope_waveform` `phase` | **radians** |
| `pulse_dt`, `pwr_levels` in the Section 3 example | **dimensionless** (pedagogical) |
| Waveform times for real experiments | **seconds** |

To convert Hz to angular frequency, always multiply by `2 * np.pi`, e.g.
`2 * np.pi * 140.0` for a 140 Hz coupling. Dropping the `-1j` or the `2*pi`
is the leading cause of wrong results.

---

## 10. Troubleshooting

**Symptom: fidelity stalls at a low value immediately (`n_iter` is tiny)**
- Check that generators are anti-Hermitian, i.e. `-1j * Ix()`. Without the
  `1j` the gradients tangle and progress stalls early.
- Check that states are normalised with `normalise_hs(...)`.
- Make sure the pulse can physically reach the target: try increasing
  `pulse_dt`, the step count, or `pwr_levels`.

**Symptom: `pwr_levels length ... must match operator count` error**
- Make the number of entries in `operators` equal to the number in
  `pwr_levels`.

**Symptom: `waveform must have shape (n_steps, n_channels)` error**
- The optimisation input waveform has **time on the rows, channels on the
  columns**. Check the `(steps, channels)` order.

**To run the pure NumPy path without Rust for comparison**
```bash
OPTIMALCONTROL_DISABLE_RUST=1 python3 my_first_pulse.py
```

**Lint and type checks**
```bash
ruff check .
mypy optimalcontrol
```

**When stuck**: the working code in `examples/` is the most reliable usage
reference.

---

## 11. Where to go next

- `docs/guide_spin_system.md` — building spin systems
- `docs/guide_states.md` — states and fidelity
- `docs/guide_operators.md` — operator utilities
- `docs/guide_grape.md` — GRAPE in depth (Hilbert/Liouville paths)
- `docs/guide_rope_crop.md` — ROPE/CROP theory
- `docs/guide_ensembles_penalties.md` — ensembles and penalties
- `docs/guide_paper_figures.md` — reproducing paper figures
- `docs/spinach_mapping.md` — mapping to the Spinach API

If you made it this far, you have already (1) installed the package,
(2) optimised your first GRAPE pulse, (3) exported it to files, and
(4) run the bundled examples. Congratulations!
