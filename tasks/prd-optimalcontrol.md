# PRD: optimalcontrol — Python Optimal Control Package for NMR

## Introduction

`optimalcontrol` is a Python package implementing analytical and numerical optimal control methods for NMR spin dynamics. It targets NMR and physics researchers who need to design pulse sequences that maximize transfer efficiency in the presence of relaxation. The package reproduces the ROPE (Relaxation-Optimized Pulse Element) and CROP (Cross-correlated Relaxation-Optimized Pulse) analytical theories from Khaneja et al. (2003 JMR, 2003 PNAS), and provides a full numerical GRAPE (Gradient Ascent Pulse Engineering) optimizer. The API preserves the same concepts and, where practical, the same function signatures as the Spinach MATLAB optimal control module.

Primary source material:
- `refdoc/1-s2.0-S109078070300003X-main.pdf`: Khaneja et al., JMR 162 (2003) 311–319 (ROPE)
- `refdoc/khaneja-et-al-2003-boundary-of-quantum-evolution-under-decoherence.pdf`: Khaneja, Luy, Glaser, PNAS 100 (2003) 13162–13166 (CROP)
- Spinach optimal control module: https://spindynamics.org/wiki/index.php?title=Optimal_control_module
- Spinach reference functions: `grape_xy.m`, `fminnewton.m`, `dirdiff.m`

---

## Goals

- Provide a self-contained Python package that NMR researchers can install and use without a MATLAB/Spinach license.
- Implement analytical ROPE and CROP formulas as exact, testable reference solutions.
- Implement numerical GRAPE gradient optimization with ensemble, penalty, and distortion support.
- Maintain Spinach-compatible API concepts (drift operators, control operators, source/target states, `ControlProblem` structure, `grape_xy` entry point) so Spinach users can migrate with minimal conceptual overhead.
- Provide paper-reproduction examples that recreate the main curves from both 2003 papers.
- Pass finite-difference gradient checks and reproduce known analytical limiting cases.
- Deliver a release-quality package with lint, type checking, and a documented test suite.

---

## User Stories

### US-001: Project Skeleton and Source Traceability
**Description:** As a researcher, I want a properly structured Python package with clear links between code and the source papers so I can trust the implementation and contribute changes confidently.

**Acceptance Criteria:**
- [ ] Package directory structure: `optimalcontrol/`, `tests/`, `examples/`, `docs/`
- [ ] `pyproject.toml` with runtime deps (`numpy`, `scipy`) and dev deps (`pytest`, `ruff`, `mypy`, `matplotlib`)
- [ ] `README.md` describing project goal, source references, and local lint/typecheck/test commands
- [ ] `docs/source_notes.md` summarizing what each PDF contributes
- [ ] `docs/spinach_mapping.md` mapping Spinach MATLAB concepts to Python modules
- [ ] `docs/equations.md` documenting ROPE/CROP equations and notation conventions
- [ ] Notation policy for Hz vs rad/s defined and documented at all conversion points
- [ ] State-vector normalization policy and fidelity conventions documented
- [ ] `optimalcontrol/__init__.py` with public API placeholders
- [ ] `tests/test_imports.py` verifies package import and version
- [ ] Source cross-reference tags in docs for paper equations and wiki concepts

### US-002: Linear Algebra and Operator Core
**Description:** As a developer building spin system models, I need a complete set of spin operators, tensor-product helpers, and Liouville-space utilities so I can construct any Hamiltonian or Liouvillian from first principles.

**Acceptance Criteria:**
- [ ] Pauli/spin-1/2 operators: `Ix`, `Iy`, `Iz`, `Ip`, `Im`, `E`
- [ ] Tensor-product helper for multi-spin operators
- [ ] Operator placement helper: place a single-spin operator at spin index `i`
- [ ] Multi-operator product builder (e.g., `2IzSz`)
- [ ] Hilbert-space commutator `comm(A, B)`
- [ ] Liouville vectorization and inverse conversion
- [ ] Left and right multiplication superoperators
- [ ] Liouvillian commutator superoperator `-1j*(L(A)-R(A))`
- [ ] Double-commutator relaxation superoperator `[F,[F,rho]]`
- [ ] Lindblad dissipator builder for general `F_k`, coefficient matrix `a_kl`
- [ ] Dense/sparse type dispatch utilities
- [ ] Shape and Hermiticity validation helpers
- [ ] Tests: spin operator commutation relations, tensor-product placement, Liouville vectorization round trip, commutator superoperator action, double-commutator relaxation
- [ ] Typecheck passes

### US-003: Spin System Model
**Description:** As a researcher, I want to define a two-spin heteronuclear system with J coupling, chemical shifts, and relaxation rates so I can build Hamiltonians and Liouvillians matching the paper's setup.

**Acceptance Criteria:**
- [ ] `Spin` dataclass with isotope label, gyromagnetic ratio, and channel
- [ ] `Coupling` dataclass for scalar J coupling in Hz
- [ ] `RelaxationRates` dataclass for DD, CSA, and DD-CSA cross-correlation rates
- [ ] `SpinSystem` dataclass with spins, couplings, shifts, relaxation rates, and basis mode
- [ ] Two-spin heteronuclear system factory for paper examples
- [ ] Drift Hamiltonian builder for scalar couplings
- [ ] Chemical-shift Hamiltonian builder
- [ ] Control operator builder by channel and axis
- [ ] Relaxation Liouvillian builder without cross-correlation
- [ ] Relaxation Liouvillian builder with DD-CSA cross-correlation
- [ ] Total generator builder: drift + relaxation + controls
- [ ] Validation for resonance/on-resonance assumptions used by ROPE/CROP
- [ ] Tests: J-coupling Hamiltonian terms, control operators on I and S spins, relaxation rates `kI = kDD + kCSA_I`, cross-correlated rates `ka`, `kc`, `ka_prime`, `kc_prime`
- [ ] Typecheck passes

### US-004: States, Targets, and Fidelity
**Description:** As a researcher, I want to construct product-operator initial and target states and evaluate transfer fidelity so I can set up ROPE/CROP and GRAPE optimization problems.

**Acceptance Criteria:**
- [ ] Product-operator state construction by labels: `Ix`, `Iz`, `2IySz`, etc.
- [ ] Single-transition operators: `IzSalpha`, `IzSbeta`, `IxSbeta`
- [ ] State normalization by 2-norm / Hilbert-Schmidt norm
- [ ] Overlap fidelity (real part, imaginary part, absolute-square)
- [ ] Multi-source/multi-target averaged fidelity
- [ ] Optional weighted fidelity over source-target pairs
- [ ] Prefix transformation hook for initial states
- [ ] Suffix transformation hook for target states
- [ ] Dead-time drift propagation before final overlap
- [ ] Tests: paper transfers `Ia -> 2IbSc`, `Ia -> Sb`, `Iz -> 2IzSz`; source-target cell-array equivalent; prefix/suffix/dead-time hooks
- [ ] Typecheck passes

### US-005: Analytical ROPE Module
**Description:** As a researcher, I want to compute ROPE analytical efficiency curves, finite-time controls, and hard-pulse approximations so I can reproduce the JMR 2003 results without running numerical optimization.

**Acceptance Criteria:**
- [ ] Relative relaxation parameter `n = kI / J`
- [ ] Unconstrained ROPE efficiency `g = sqrt(1+n^2) - n`
- [ ] INEPT efficiency curve `exp(-pi*k*t)*sin(pi*J*t)` and optimal time `t* = arccot(n)/(pi*J)`
- [ ] ROPE gain over INEPT
- [ ] In-phase ROPE efficiency `g_in` for `Ix -> Sx`
- [ ] Refocused INEPT comparison efficiency
- [ ] Optimal return function `V(r1,r2) = sqrt(g^2*r1^2 + r2^2)`
- [ ] Unconstrained optimal control relation `u2/u1 = g*r1/r2` with singular-case handling
- [ ] Optimal trajectory invariant `expect(2IySz)/expect(Ix) = g`
- [ ] Finite-time critical duration `Tcrit = arccot(2n)/(pi*J)`
- [ ] Finite-time `j(s)` function from Appendix B
- [ ] Equation solver for switching time `s` given total duration `T`
- [ ] Finite-time angles `h1`, `h2` and efficiency `g_T`
- [ ] Phase-I, phase-II, phase-III control waveforms
- [ ] RF amplitude conversion and hard-pulse boundary flip-angle calculation
- [ ] Finite-time waveform sampler returning times, controls, amplitudes, and phases
- [ ] Tests: `n=0` no-relaxation limit; paper example `n=1` finite-time values; finite-time efficiency approaches `g` as `T` grows; `T <= Tcrit` reduces to INEPT controls
- [ ] Typecheck passes

### US-006: Analytical CROP and Decoherence Boundary Module
**Description:** As a researcher, I want to compute CROP efficiency limits and pulse parameters for systems with DD-CSA cross-correlation so I can reproduce the PNAS 2003 decoherence boundary results.

**Acceptance Criteria:**
- [ ] `zeta = sqrt((ka^2-kc^2)/(J^2+kc^2))` with domain checks
- [ ] CROP efficiency `eta = sqrt(1+zeta^2) - zeta` and primed variant for spin S rates
- [ ] Physical transfer limits: `Iz -> 2IzSz`, `2IzSz -> Sz`, `Iz -> Sz` as `eta*eta_prime`, single-transition transfer `sqrt(eta^2 + eta_prime^2)`
- [ ] Limiting cases: `kc=0` reduces to ROPE-like expression; `kc/ka -> 1` approaching lossless transfer
- [ ] Single-transition decomposition helpers and slowly relaxing multiplet component identification
- [ ] CROP pulse parameter container: amplitude, irradiation frequency, truncation window
- [ ] Truncated CROP waveform generator
- [ ] CROP robustness sweep over `ka/J` and `kc/ka`
- [ ] Tests: table of physical limits from PNAS paper; monotonic improvement as `kc/ka` increases; no efficiency above 1 except numerical tolerance; regression values `ka/J = 0.6`, `1.1`, `kc/ka = 0.75`
- [ ] Typecheck passes

### US-007: Propagation and GRAPE Cost Functions
**Description:** As a researcher, I want to run GRAPE optimization with forward/backward propagation, analytical gradients, and Hessian options so I can numerically optimize pulse sequences beyond what analytical theory provides.

**Acceptance Criteria:**
- [ ] `ControlProblem` dataclass matching Spinach `control` structure: `drifts`, `operators`, `rho_init`, `rho_targ`, `pulse_dt`, `pwr_levels`, `freeze`, and optional fields (fidelity mode, offsets, offset operators, phase cycle, basis, penalties, plotting, checkpoint)
- [ ] Waveform shape validation for Cartesian XY controls
- [ ] Basis-coefficient expansion for smooth waveform bases
- [ ] Freeze-mask application to waveform updates and gradients
- [ ] Per-slice Hamiltonian/Liouvillian assembly
- [ ] Forward propagation cache and backward adjoint propagation cache over pulse slices
- [ ] Final-state evaluation for all source-target pairs
- [ ] `grape_xy` function returning fidelity only (Spinach-compatible entry point)
- [ ] Directional derivative of matrix exponential using auxiliary matrix method
- [ ] Gradient accumulation over slices and control channels
- [ ] Exact Hessian path for small dense systems
- [ ] Liouville-space GRAPE variant
- [ ] Hilbert-space GRAPE variant for closed systems
- [ ] Phase-only GRAPE, amplitude-phase to XY conversion, curvilinear parameterization adapters
- [ ] Tests: finite-difference gradient checks for two-spin systems; Hessian finite-difference for one-spin; freeze preservation; basis expansion dimensions
- [ ] Typecheck passes

### US-008: Ensemble Control
**Description:** As a researcher, I want to optimize pulses that are robust over distributions of spin systems (multiple drift generators, power levels, offsets, phase cycles) so I can design pulses that perform well across realistic experimental conditions.

**Acceptance Criteria:**
- [ ] Ensemble expansion over: source-target pairs, multiple drift generators, control power levels, offsets, phase-cycle rows
- [ ] Cartesian product ensemble builder
- [ ] Correlated ensemble modes: `rho_match` and `rho_drift`
- [ ] Ensemble fidelity and gradient aggregation
- [ ] Optional multiprocessing/joblib backend abstraction
- [ ] Deterministic serial backend for tests
- [ ] Tests: ensemble size calculation; non-combinatorial correlated modes; offset operator inclusion; phase-cycle transformation consistency
- [ ] Typecheck passes

### US-009: Penalties, Distortions, and Constraints
**Description:** As a researcher, I want to add waveform penalties and amplifier/filter distortion models to GRAPE so I can enforce hardware constraints and account for pulse transducer imperfections.

**Acceptance Criteria:**
- [ ] Penalties: norm-square (`NS`), spillout norm-square Cartesian (`SNS`), spillout norm-square amplitude (`SNSA`), derivative norm-square (`DNS`)
- [ ] Penalty weights, penalty values, gradients, and Hessians where practical
- [ ] Separated reporting of simulation fidelity vs penalties
- [ ] Lower and upper waveform bounds; amplitude clipping helper for diagnostics
- [ ] Distortion models: no-op, tanh amplifier compression, root amplifier compression, single-pole filter, single-zero filter, transfer-matrix
- [ ] Distortion derivative hooks for GRAPE gradients
- [ ] Tests: each penalty value and gradient; distortion identity and simple filter responses
- [ ] Typecheck passes

### US-010: Optimizers
**Description:** As a researcher, I want multiple optimizer backends (gradient ascent, LBFGS, Newton-Raphson) with convergence controls and checkpoint/resume so I can run long optimizations reliably and choose the best algorithm for my problem size.

**Acceptance Criteria:**
- [ ] Optimizer result dataclass: final waveform, counts, convergence reason, history
- [ ] Cubic/bracketed line search
- [ ] Gradient ascent baseline optimizer
- [ ] LBFGS memory state, inverse-Hessian update, and LBFGS-GRAPE optimizer path
- [ ] Newton-Raphson path with Hessian regularization and RFO-style regularized Newton option
- [ ] Convergence checks: step norm `tol_x`, gradient norm `tol_g`, max iterations, max evaluations
- [ ] Checkpoint save and resume
- [ ] Verbose iteration table matching Spinach diagnostics output
- [ ] Tests: quadratic toy objective; LBFGS improves GRAPE fidelity on small state-transfer; checkpoint/resume equivalence
- [ ] Typecheck passes

### US-011: Input, Output, and Diagnostics
**Description:** As a researcher, I want to export and import optimized waveforms, analyze state trajectories, and generate diagnostic plots so I can inspect optimization results and share pulse sequences with colleagues or load them onto spectrometers.

**Acceptance Criteria:**
- [ ] Waveform container with channels, units, time grid, metadata, and source problem hash
- [ ] CSV and JSON waveform export; JSON waveform import
- [ ] FAPT-like frequency-amplitude-phase-time import
- [ ] Bruker pulse export stub (documented limitations)
- [ ] JCAMP-DX pulse import stub (documented limitations)
- [ ] Heterodyne transformation helper
- [ ] Trajectory analysis: state populations, local spin expectation values, coherence order, correlation order
- [ ] Robustness histogram data generation
- [ ] Spectrogram data generation for channel pairs
- [ ] Plotting functions: XY controls, amplitude/phase controls, ROPE/CROP efficiency curves, state trajectories
- [ ] Tests: waveform import/export round trips; plotting functions returning figure objects without display
- [ ] Typecheck passes

### US-012: Paper Reproduction Examples
**Description:** As a researcher, I want runnable example scripts that reproduce the main figures from both 2003 papers so I can verify the implementation is correct and use the examples as templates for my own problems.

**Acceptance Criteria:**
- [ ] ROPE vs INEPT efficiency as a function of `n`
- [ ] ROPE gain curve for `Ia -> 2IbSc`
- [ ] In-phase transfer gain curve
- [ ] Finite-time ROPE efficiency vs total transfer time
- [ ] Finite-time ROPE controls for `n=1`
- [ ] Sodium-formate-like parameters: `J=193 Hz`, `T2=1.4 ms`
- [ ] Finite-time ROPE waveform to hard-pulse approximation
- [ ] CROP efficiency vs `ka/J` for several `kc/ka` values
- [ ] CROP vs INEPT/CRIPT/CRINEPT placeholder comparison
- [ ] CROP truncated amplitude and irradiation frequency plots
- [ ] CROP parameters `ka/J=0.6`, `1.1`, `kc/ka=0.75`
- [ ] `kc/ka -> 1` decoherence-free limit example
- [ ] Expected-data snapshots for all paper reproduction examples
- [ ] Tests that all examples run in headless mode
- [ ] Typecheck passes

### US-013: Integration Milestones
**Description:** As a developer, I want end-to-end integration tests that verify all major subsystems work together so I can confirm the full pipeline from spin system definition to optimized waveform export is functional.

**Acceptance Criteria:**
- [ ] Operator core + state builders + analytical ROPE exposed in one public API
- [ ] CROP analytical limits integrated with two-spin system factory
- [ ] `ControlProblem` validation integrated with GRAPE cost functions
- [ ] Penalties integrated into `grape_xy` fidelity/gradient output
- [ ] Ensemble expansion integrated into `grape_xy`
- [ ] Optimizer integrated with `grape_xy` and checkpointing
- [ ] Trajectory diagnostics integrated with optimized waveforms
- [ ] Waveform export integrated with examples
- [ ] End-to-end: analytical ROPE waveform beats INEPT for `n=1`
- [ ] End-to-end: numerical GRAPE improves random guess for two-spin transfer
- [ ] End-to-end: ensemble power robustness runs without crash and reports histogram
- [ ] End-to-end: optimizer resume reproduces uninterrupted result within tolerance
- [ ] End-to-end: exported waveform can be imported and replayed
- [ ] Documentation pages: spin system definition, initial/target states, drift/control operators, analytical ROPE/CROP, numerical GRAPE, ensembles/penalties, paper figure reproduction
- [ ] Typecheck passes

### US-014: Validation, Quality, and Release Preparation
**Description:** As a developer, I want a release-quality package with linting, type checking, docstring coverage, numerical tolerance policies, performance benchmarks, and semantic versioning so I can confidently publish and maintain the package.

**Acceptance Criteria:**
- [ ] Numerical tolerance policy documented for dense and sparse paths
- [ ] Unit tests for all documented public functions
- [ ] Regression tests for all analytical formulas
- [ ] Finite-difference gradient test suite with seeded random problems
- [ ] Performance benchmark: matrix exponential derivative methods
- [ ] Performance benchmark: ensemble scaling
- [ ] `ruff` lint configuration passing with zero issues
- [ ] `mypy` configuration passing for public dataclasses and APIs
- [ ] Docstring coverage for all public functions
- [ ] Error-message tests for invalid dimensions and invalid physical parameters
- [ ] Reproducibility controls for random initial guesses
- [ ] Semantic versioning policy and changelog
- [ ] Full test suite runs to baseline with pass/fail record
- [ ] Internal milestones tagged: analytical ROPE/CROP complete, GRAPE dense complete, ensemble/penalty complete, examples/docs complete
- [ ] Typecheck passes

---

## Functional Requirements

- **FR-1:** The package must install via `pip install .` with only `numpy` and `scipy` as runtime dependencies.
- **FR-2:** All public functions that have Spinach MATLAB equivalents (`grape_xy`, operator builders, fidelity evaluators) must use the same argument names and semantics as the Spinach counterparts.
- **FR-3:** The `grape_xy` function must accept a `ControlProblem` dataclass and return at minimum a fidelity scalar, matching the Spinach calling convention.
- **FR-4:** Analytical ROPE functions must reproduce `g = sqrt(1+n^2) - n` for unconstrained transfer and pass finite-time limit tests from Appendix B of the JMR paper.
- **FR-5:** Analytical CROP functions must reproduce the physical transfer limits table from Table 1 of the PNAS paper within documented numerical tolerance.
- **FR-6:** GRAPE gradients must pass finite-difference checks on at least a one-spin and a two-spin system with tolerance ≤ 1e-6 relative error.
- **FR-7:** Ensemble dimensions must be independently configurable over drift generators, power levels, offsets, and phase cycles in any combination.
- **FR-8:** Waveform export (CSV, JSON) and import must round-trip without loss of amplitude, phase, or time grid.
- **FR-9:** All example scripts must run in headless mode (no display required) and complete without error.
- **FR-10:** The optimizer must support checkpoint save and resume that reproduces an uninterrupted run within documented tolerance.
- **FR-11:** Frequency units must be consistently Hz throughout the public API; internal rad/s conversions must occur at documented boundaries.
- **FR-12:** All public dataclasses must pass `mypy` strict type checking.
- **FR-13:** The package must define and enforce state normalization conventions (Hilbert-Schmidt norm) at all fidelity evaluation points.
- **FR-14:** Distortion models must expose derivative hooks so GRAPE gradients remain correct when distortions are enabled.

---

## Non-Goals

- No GUI or interactive widgets (all output is figures, CSV, or JSON).
- No MATLAB interoperability layer — the package does not call or wrap Spinach.
- No multi-quantum or higher-spin (>1/2) operators in the initial release; two-spin-1/2 systems are the primary target.
- No Bruker or JCAMP-DX format full implementation — stubs with documented limitations only.
- No distributed computing or GPU acceleration — CPU-only with optional joblib parallelism.
- No automatic differentiation framework (e.g., JAX, PyTorch) — gradients are derived analytically.
- No web service or REST API.
- No automatic paper figure layout — examples produce data and figure objects; publication formatting is left to the user.

---

## Technical Considerations

- **Spinach API compatibility:** Function signatures, argument ordering, and field names in `ControlProblem` must follow Spinach conventions documented in `docs/spinach_mapping.md`. Deviations require explicit justification in that document.
- **Dense vs sparse dispatch:** All operator builders must support both dense (`numpy.ndarray`) and sparse (`scipy.sparse`) matrices, selected by a `basis` flag on `SpinSystem`.
- **Matrix exponential:** Use `scipy.linalg.expm` for dense systems; directional derivative uses the auxiliary matrix method from Khaneja et al. to avoid finite-difference error in gradients.
- **Numerical stability:** Liouville-space operators for small systems (≤ 4 spins) are dense; sparse paths are reserved for larger systems.
- **Testing framework:** `pytest` with `numpy.testing.assert_allclose` for all numerical comparisons; tolerance documented per test.
- **Code style:** `ruff` for linting, `mypy` for type checking; no external type stubs beyond `numpy` and `scipy`.
- **Examples:** Each example in `examples/` must be a self-contained script that imports only from `optimalcontrol`, `numpy`, `scipy`, and `matplotlib`.

---

## Success Metrics

- Analytical ROPE functions reproduce the no-relaxation, finite-time, and high-relaxation limiting cases from JMR 2003.
- Analytical CROP functions reproduce the `kc=0` and `kc/ka -> 1` limiting cases from PNAS 2003.
- A two-spin dense GRAPE example optimizes `Iz -> 2IzSz` from a random initial guess and improves fidelity.
- Gradient checks pass against finite differences on tiny systems with relative error ≤ 1e-6.
- All 12 paper-reproduction example scripts run headless and produce qualitatively correct curves.
- The API covers all major Spinach optimal-control concepts: drift/control operators, source-target pairs, ensemble dimensions, penalties, optimizer selection, checkpointing, and plotting diagnostics.
- Full test suite passes with zero failures on the baseline run.

---

## Open Questions

- Should `grape_xy` return a named tuple or a dataclass to carry gradient and Hessian alongside fidelity, or match the bare scalar return of Spinach exactly and expose a separate function for gradient/Hessian?
- For sparse Liouville-space systems beyond 4 spins, which `scipy.sparse.linalg` matrix exponential approximation (Padé via `expm_multiply` or Krylov) should be the default?
- Should the `rho_match` and `rho_drift` correlated ensemble modes follow Spinach naming exactly, or adopt more descriptive Python names with Spinach names as aliases?
- Are there additional NMR pulse format targets (e.g., Varian/Agilent `.rf`, Siemens `.pta`) beyond Bruker and JCAMP-DX that should be stubbed in the initial release?
