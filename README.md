# optimalcontrol

Python/Rust package for NMR spin dynamics implementing analytical ROPE/CROP theories,
numerical GRAPE optimisation, and the Seedless band/restraint front-end for isolated
spin-1/2 pulse design, with a Spinach-compatible Python API, ensemble support, and
paper-reproduction examples. GRAPE propagation, exact coherent gradients, Bloch ensemble
profiles, and the analytic spin-1/2 Seedless kernel run in a parallel native Rust
extension.

## Installation

```bash
pip install optimalcontrol
```

Python 3.10 or newer is required. Environments without a compatible prebuilt wheel also
need a stable Rust toolchain so `pip` can compile the native extension from the sdist.

## Getting started

New to the package? Follow the step-by-step beginner manual in
[`docs/user_manual.md`](docs/user_manual.md): install, design a pulse by two different
routes (GRAPE and Seedless), read the result, verify it against an independent Bloch
model, and write a shape file. The topic-specific `docs/guide_*.md` files cover each
subsystem in depth.

## Source references

- **JMR 2003 (ROPE)**: Unterbeck & Glaser, *Journal of Magnetic Resonance* 160 (2003) 88–101 — analytical optimal control for heteronuclear transfer under relaxation.
- **PNAS 2003 (CROP)**: Unterbeck & Glaser, *Proc. Natl. Acad. Sci. USA* 100 (2003) 5172–5177 — cross-correlated relaxation-optimised pulses.
- **Nat. Commun. 2025 (Seedless)**: Buchanan et al., *Nature Communications* 16, 7276 (2025) — ["Seedless: on-the-fly pulse calculation for NMR experiments"](https://doi.org/10.1038/s41467-025-61663-8); the band/restraint formalism implemented by `optimalcontrol.ocseed`.
- **Spinach**: [https://spindynamics.org/wiki/index.php?title=Main_Page](https://spindynamics.org/wiki/index.php?title=Main_Page) — MATLAB spin dynamics library whose `grape_xy` / `control` struct API this package mirrors.

## Local development commands

Install a stable Rust toolchain (`rustc` and `cargo`) first. On macOS with Homebrew:

```bash
brew install rust
```

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Lint
ruff check .

# Typecheck
mypy optimalcontrol

# Test
python3 -m pytest
```

Set `OPTIMALCONTROL_DISABLE_RUST=1` to run the NumPy/SciPy fallback for numerical
comparisons. Normal installations build and use the Rust extension automatically.

## Performance

On an Apple Silicon development machine, the native path reduced the full example
regression runtime from 20.08 s to 2.91 s (6.9x). The focused 72-slice GRAPE benchmark
improved single-member gradient evaluation by 2.2x and five-member ensemble gradient
evaluation by 3.4x. Run `python benchmarks/bench_grape_hotpath.py` and
`pytest tests/test_examples.py` to measure the local machine.

## Symmetric methyl 180 pulse with water preservation

`python -m examples.methyl_water_binary_symmetric_180` writes a Bruker shape and a
dense-grid diagnostic plot for a 1.2 GHz proton spectrometer with the carrier at water
(4.7 ppm). The 1.740 ms pulse covers methyl protons from -3 to 3 ppm, preserves water
`Iz`, uses only 0/180 degree phase, is exactly time symmetric, and is capped at 10 kHz.
It implements the inversion-quality requirement motivated by Kay's methyl-HMQC
refocusing-artifact analysis ([JBNMR 2019](https://doi.org/10.1007/s10858-019-00227-7)).

The cached candidate was validated at 2401 methyl offsets and 9 water offsets. Worst
fidelities are 0.999186 (`Ix -> Ix`), 0.999128 (`-Iy -> Iy`), 0.999098 (`Iz -> -Iz`),
and 0.999892 for water `Iz -> Iz`; the worst predicted inner artifact is 0.09027% of
the central line. It is the shortest passing point on the tested local duration grid:
1.740 ms passed while 1.735 ms failed. This is a numerical grid-search result,
not a proof of the continuous global minimum. The recorded duration audit is in
`examples/expected/methyl_water_binary_symmetric_180_duration_search.csv`.

## Versioning policy

This package follows semantic versioning: `MAJOR.MINOR.PATCH`.

- `PATCH` releases contain backwards-compatible fixes only and must not introduce breaking API,
  file-format, or numerical-contract changes.
- `MINOR` releases may add features and deprecate existing APIs; deprecations must emit warnings
  before removal.
- `MAJOR` releases may remove deprecated APIs or introduce intentional breaking changes, with
  migration notes recorded in `CHANGELOG.md`.
