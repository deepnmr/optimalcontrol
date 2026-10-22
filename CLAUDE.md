# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`optimalcontrol` (PyPI: `optimalcontrol-nmr`, import name unchanged): NMR pulse design.
Analytical ROPE/CROP theory, a Spinach-compatible GRAPE engine, and the Seedless
band/restraint front-end (`ocseed`), with hot paths in a Rust extension (`src/lib.rs` →
`optimalcontrol._rust`, built by maturin). Also ships an MCP server and a Claude Code plugin.

## Commands

Always use the project venv; system `python3` has no numpy. `.venv/bin/pip` is broken — use
`.venv/bin/python -m pip`. Python 3.11 is the floor (`requires-python = ">=3.11"`); CI also
runs 3.12.

```bash
.venv/bin/python -m pytest -q                       # full suite (~25 s); run pytest for the count
.venv/bin/python -m pytest -q tests/test_grape.py   # one file
.venv/bin/python -m pytest -q tests/test_ocseed.py::test_kernel_rejects_non_finite_inputs
.venv/bin/ruff check optimalcontrol/ tests/ examples/ benchmarks/          # lint gate (E, F, I, B, UP, NPY)
.venv/bin/ruff format --check optimalcontrol/ tests/ examples/ benchmarks/ # ruff format IS enforced
.venv/bin/python -m mypy optimalcontrol/            # strict
OPTIMALCONTROL_DISABLE_RUST=1 .venv/bin/python -m pytest -q tests/test_ocseed.py   # NumPy fallback parity
OPTIMALCONTROL_DISABLE_RUST=1 .venv/bin/python -m pytest -q --deselect tests/test_rust_accelerator.py   # full fallback suite (~12 min)
cargo fmt --check                                   # Rust formatting gate
cargo clippy --locked --all-targets -- -D warnings  # Rust lint gate, warnings denied
.venv/bin/maturin develop --release                 # rebuild the Rust extension after editing src/lib.rs
.venv/bin/python -c "from optimalcontrol.ocseed import demo; demo()"               # quick smoke: "demo ok (...)"
.venv/bin/python -m examples.methyl_water_binary_symmetric_180                     # run an example
.venv/bin/python benchmarks/bench_grape_hotpath.py
.venv/bin/python benchmarks/bench_bloch.py   # Bloch ensemble timing, both backends (OPTIMALCONTROL_DISABLE_RUST=1 for NumPy)
```

- `tests/test_rust_accelerator.py` fails under `OPTIMALCONTROL_DISABLE_RUST=1` by design (it
  tests the Rust path). Everything else must pass on both paths.
- `tests/test_examples.py` calls each listed example's `run()` once, checks that it returns
  a result, and compares to `examples/expected/<name>.npz` (key `"output"`, rtol 1e-4,
  atol 1e-12). A numerical change to an example is a snapshot change; regenerate
  deliberately, don't loosen the tolerance.
- Error message strings are tested (`tests/test_validation.py` etc.); `_validation.py` says
  keep them stable.
- `.github/workflows/ci.yml` runs on pull requests and pushes to `master`, all on Ubuntu with
  stable Rust. The `checks` job is a Python 3.11/3.12 matrix: Ruff, `ruff format --check`,
  mypy, Clippy with warnings denied, `cargo fmt --check`, `pip check`, and the full native
  suite. A separate `fallback` job (3.12) runs the whole suite under
  `OPTIMALCONTROL_DISABLE_RUST=1` with `tests/test_rust_accelerator.py` deselected. The
  `package` job builds the sdist and wheel with maturin, installs the wheel, runs an import
  smoke test, and checks the sdist for leaked repo-only files.

## Architecture

**Layers (bottom → top):**

1. `operators.py`, `states.py`, `spin_system.py` — spin-1/2 operators, product-operator
   labels (`state_from_label("2IzSz", n)`), Liouville superoperators. Column-major `vec()`
   convention: `vec(A X B) = (Bᵀ ⊗ A) vec(X)`; `L_op(A) = I ⊗ A`, `R_op(A) = Aᵀ ⊗ I`.
   Anything that mixes conventions breaks Liouville propagation silently.
2. `_problem.py` defines the `ControlProblem` dataclass (mirrors Spinach's `control`
   struct), its validation and `_hash_control_problem`; `_propagation.py` holds waveform
   freezing and first-order state propagation; `_hessian.py` holds the second-order
   propagation behind `grape_hessian`. `grape.py` re-exports `ControlProblem` and owns
   `grape_xy` / `grape_xy_and_gradient` / `grape_hessian`. `basis` is `"dense"`,
   `"liouville"` or `"hilbert"`. Coherent problems (all `1j*generator` Hermitian, checked by
   `_accelerator._is_anti_hermitian`) take the eigendecomposition path; dissipative ones use
   `expm`.
3. `ensemble.py` — any of `len(drifts)>1`, RF power levels, `offsets`+`offset_operators`,
   or `phase_cycle` makes `_has_ensemble_axes` true; `cartesian_product_ensemble` expands
   to single problems and results are averaged. Penalties (`penalties.py`) are stripped
   before member evaluation and subtracted once from the mean.
4. `_accelerator.py` — the only place that touches `_rust`. Every public entry point
   (`vector_fidelity`, `vector_value_gradient`, `problem_vector_*`) returns `None` when the
   Rust path can't handle the problem, and callers fall through to NumPy. Validation
   happens Python-side *before* dispatch so both paths raise identically; the seedless
   Rust kernels in particular do not check finiteness themselves (the GRAPE ones do).
5. `optimizers.py` — `lbfgs_grape`, `newton_raphson`, and `run_grape(cp, wfm0, method=)`
   which returns `(io.Waveform, OptimResult)`. Its pieces live in `_checkpoint.py`
   (`OptimResult` and checkpoint persistence), `_line_search.py` (strong-Wolfe cubic line
   search), `_lbfgs.py` (L-BFGS memory and direction) and `_newton.py` (Newton/RFO step
   helpers). Checkpoints are keyed by `_problem._hash_control_problem` (re-exported from
   `io`), so changing a `ControlProblem` field invalidates them.
6. `ocseed.py` + `_seedless_kernel.py` — Seedless: `SeedlessSpec` with `Band`s
   (`universal` / `s2s` / `xycite` / `suppress`), constant-amplitude phase-only
   optimisation over an (offset × B1) ensemble. `fast=True` (default) uses the analytic 2×2
   scaled-unitary kernel (Rust or NumPy); `fast=False` routes through the 4×4 Liouville
   GRAPE core as a cross-check. Ensemble members are offset-major (`offset[m // n_b1]`,
   `b1[m % n_b1]`). `bloch.py` is the independent forward model used for verification only.
   The NumPy fallback of `propagate_bloch_ensemble` is a vectorised Rodrigues update that
   mirrors `src/lib.rs` `rotate_bloch` (same expression order, agrees to ~1e-13), with one
   `_rotate` call per step over the whole ensemble.
7. `io.py` — `Waveform` container, CSV/JSON/Bruker JCAMP export and import. It's a facade
   over `_io_common.py` (the `Waveform` class and path/format helpers), `_io_bruker.py`
   (Bruker shape export) and `_io_jcamp.py` (JCAMP-DX and FAPT import).
   `plotting.py` is optional (matplotlib), `analysis.py` gives trajectories/spectrograms.
8. `mcp_server.py` — mcp SDK **2.0** (`from mcp.server import MCPServer`; `FastMCP` no
   longer exists). Two tools wrap the Seedless route; docstrings are the user-facing
   contract (units: ppm for bands/carrier, Hz for RF, seconds for duration).

Example helpers shared across scripts live in `examples/_methyl_water_common.py` (the
methyl/water search, refinement and plotting code), `examples/_figures.py` (`save_figure`
writes PNGs with version-free metadata; `quantise` rounds Bloch-derived curves to 12
decimals at the plot call so both backends render identical PNGs, and library return values
are never rounded) and `examples/_phase_waveforms.py` (phase to x/y waveform conversion).

The public import paths are frozen by `tests/test_public_surface.py`: every pre-refactor
import must still resolve, so move code into private modules and re-export it.

**Numerical contract:** Rust == NumPy fallback == general GRAPE engine to ~1e-15, gradients
checked by finite differences to ~1e-9. Any change to a kernel must keep
`OPTIMALCONTROL_DISABLE_RUST=1` parity and the FD tests green.

**Rust side:** `src/lib.rs` is a single pyo3 module (crate name `optimalcontrol-rust`,
version must match `pyproject.toml` and `optimalcontrol/__init__.py`). Exposed functions:
`grape_fidelity_vectors`, `grape_value_gradient_vectors`, `bloch_ensemble`,
`seedless_pair_value_gradient`, `seedless_suppress_perstep`.

## Release and repo conventions

- Semantic versioning (see README "Versioning policy"). Bump all 6 version sites together:
  `pyproject.toml`, `Cargo.toml`, `Cargo.lock`, `optimalcontrol/__init__.py`,
  `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`; promote CHANGELOG `Unreleased`;
  tag `vX.Y.Z`; `maturin build --release --sdist -o dist`; `twine upload`. CI checks changes;
  publishing remains manual.
- `.claude-plugin/plugin.json` pins `>=0.5.0` and carries its own `version`.
- Pushing to `deepnmr/optimalcontrol` needs `gh auth switch -u deepnmr`; switch back to
  `dleess` afterwards.
- `scripts/ralph/` and `dev/` are finished agent-loop scaffolding, excluded from the sdist.
  Don't take `scripts/ralph/CLAUDE.md` as instructions for this repo.
- `HANDOFF.md` (gitignored) is the running session handoff; read it first if present.
