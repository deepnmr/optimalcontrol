# Optimal Control Programming Plan

## Scope

This plan turns the two PDFs in `refdoc/` and the Spinach optimal control module reference into a Python implementation plan. The MATLAB/Spinach reference is used as the functional model, but the target architecture is a Python package with NumPy/SciPy style APIs, tests, examples, and later integration points.

Primary source material:

- `refdoc/1-s2.0-S109078070300003X-main.pdf`: Khaneja et al., "Optimal control of spin dynamics in the presence of relaxation", JMR 162 (2003) 311-319.
- `refdoc/khaneja-et-al-2003-boundary-of-quantum-evolution-under-decoherence.pdf`: Khaneja, Luy, Glaser, "Boundary of quantum evolution under decoherence", PNAS 100 (2003) 13162-13166.
- Spinach optimal control module: https://spindynamics.org/wiki/index.php?title=Optimal_control_module
- Spinach supporting function references: `grape_xy.m`, `fminnewton.m`, `dirdiff.m`.

## Implementation Assumptions

- Build a Python package named `optimalcontrol`.
- Use `numpy` for dense arrays and `scipy.linalg` / `scipy.sparse` for matrix exponentials and sparse operators.
- Start with dense two-spin and small-system support, then add sparse Liouville-space paths.
- Treat the analytical ROPE/CROP formulas as first-class modules and use them as reference tests for the numerical GRAPE implementation.
- Keep the API independent from Spinach while preserving the same concepts: drift operators, control operators, source/target states, ensembles, penalties, waveform guesses, fidelity/gradient/Hessian evaluators, and optimizers.
- Defer GUI work. Use command-line examples, plots, and generated data files first.

## Work Breakdown

### Phase 1: Project Skeleton And Source Traceability

- [ ] T001 Create Python package directories: `optimalcontrol/`, `tests/`, `examples/`, `docs/`.
- [ ] T002 Create `pyproject.toml` with package metadata and dependency declarations.
- [ ] T003 Add runtime dependencies: `numpy`, `scipy`.
- [ ] T004 Add development dependencies: `pytest`, `ruff`, `mypy`, `matplotlib`.
- [ ] T005 Add a minimal `README.md` describing project goal and source references.
- [ ] T006 Add `docs/source_notes.md` summarizing what each PDF contributes.
- [ ] T007 Add `docs/spinach_mapping.md` mapping Spinach MATLAB concepts to Python modules.
- [ ] T008 Add `docs/equations.md` for ROPE/CROP equations and notation conventions.
- [ ] T009 Define notation policy for Hz vs rad/s and document all conversion points.
- [ ] T010 Define state-vector normalization policy and document fidelity conventions.
- [ ] T011 Add package-level `__init__.py` with public API placeholders.
- [ ] T012 Add CI-style local commands in `README.md`: lint, type check, test.
- [ ] T013 Add `tests/test_imports.py` to verify package import and version.
- [ ] T014 Add source cross-reference tags in docs for paper equations and wiki concepts.

### Phase 2: Linear Algebra And Operator Core

- [ ] T015 Implement Pauli/spin-1/2 single-spin operators `Ix`, `Iy`, `Iz`, `Ip`, `Im`, `E`.
- [ ] T016 Implement tensor-product helper for multi-spin operators.
- [ ] T017 Implement operator placement helper: place a single-spin operator at spin index `i`.
- [ ] T018 Implement multi-operator product builder such as `2IzSz`.
- [ ] T019 Implement Hilbert-space commutator `comm(A, B)`.
- [ ] T020 Implement Liouville vectorization convention and inverse conversion.
- [ ] T021 Implement left multiplication superoperator.
- [ ] T022 Implement right multiplication superoperator.
- [ ] T023 Implement Liouvillian commutator superoperator `-1j*(L(A)-R(A))`.
- [ ] T024 Implement double-commutator relaxation superoperator `[F,[F,rho]]`.
- [ ] T025 Implement Lindblad dissipator builder for general `F_k`, coefficient matrix `a_kl`.
- [ ] T026 Implement dense/sparse type dispatch utilities.
- [ ] T027 Add shape and Hermiticity validation helpers.
- [ ] T028 Add tests for spin operator commutation relations.
- [ ] T029 Add tests for tensor-product placement and two-spin product operators.
- [ ] T030 Add tests for Liouville vectorization round trip.
- [ ] T031 Add tests comparing commutator superoperator action to direct matrix commutator.
- [ ] T032 Add tests for double-commutator relaxation on simple density matrices.

### Phase 3: Spin System Model

- [ ] T033 Define `Spin` dataclass with isotope label, optional gyromagnetic ratio, and channel.
- [ ] T034 Define `Coupling` dataclass for scalar J coupling in Hz.
- [ ] T035 Define `RelaxationRates` dataclass for DD, CSA, and DD-CSA cross-correlation rates.
- [ ] T036 Define `SpinSystem` dataclass with spins, couplings, shifts, relaxation rates, and basis mode.
- [ ] T037 Implement two-spin heteronuclear system factory for paper examples.
- [ ] T038 Implement drift Hamiltonian builder for scalar couplings.
- [ ] T039 Implement chemical-shift Hamiltonian builder.
- [ ] T040 Implement control operator builder by channel and axis.
- [ ] T041 Implement relaxation Liouvillian builder without cross-correlation.
- [ ] T042 Implement relaxation Liouvillian builder with DD-CSA cross-correlation.
- [ ] T043 Implement total generator builder: drift plus relaxation plus controls.
- [ ] T044 Add validation for resonance/on-resonance assumptions used by ROPE/CROP.
- [ ] T045 Add tests for two-spin J-coupling Hamiltonian terms.
- [ ] T046 Add tests for control operators on I and S spins.
- [ ] T047 Add tests for relaxation rates `kI = kDD + kCSA_I` and `kS = kDD + kCSA_S`.
- [ ] T048 Add tests for cross-correlated rates `ka`, `kc`, `ka_prime`, `kc_prime`.

### Phase 4: States, Targets, And Fidelity

- [ ] T049 Implement product-operator state construction by labels such as `Ix`, `Iz`, `2IySz`.
- [ ] T050 Implement single-transition operators such as `IzSalpha`, `IzSbeta`, `IxSbeta`.
- [ ] T051 Implement state normalization by 2-norm / Hilbert-Schmidt norm.
- [ ] T052 Implement overlap fidelity real part.
- [ ] T053 Implement overlap fidelity imaginary part.
- [ ] T054 Implement absolute-square fidelity.
- [ ] T055 Implement multi-source/multi-target averaged fidelity.
- [ ] T056 Implement optional weighted fidelity over source-target pairs.
- [ ] T057 Implement prefix transformation hook for initial states.
- [ ] T058 Implement suffix transformation hook for target states.
- [ ] T059 Implement dead-time drift propagation before final overlap.
- [ ] T060 Add tests for paper transfers: `Ia -> 2IbSc`, `Ia -> Sb`, `Iz -> 2IzSz`.
- [ ] T061 Add tests for source-target cell-array equivalent behavior.
- [ ] T062 Add tests for prefix/suffix/dead-time hooks.

### Phase 5: Analytical ROPE Module

- [ ] T063 Implement relative relaxation parameter `n = kI / J`.
- [ ] T064 Implement unconstrained ROPE efficiency `g = sqrt(1+n^2) - n`.
- [ ] T065 Implement INEPT efficiency curve `exp(-pi*k*t)*sin(pi*J*t)`.
- [ ] T066 Implement INEPT optimal time `t* = arccot(n)/(pi*J)`.
- [ ] T067 Implement INEPT maximum efficiency from the analytical time.
- [ ] T068 Implement ROPE gain over INEPT.
- [ ] T069 Implement in-phase ROPE efficiency `g_in` for `Ix -> Sx`.
- [ ] T070 Implement refocused INEPT comparison efficiency.
- [ ] T071 Implement optimal return function `V(r1,r2) = sqrt(g^2*r1^2 + r2^2)`.
- [ ] T072 Implement unconstrained optimal control relation `u2/u1 = g*r1/r2` with singular-case handling.
- [ ] T073 Implement optimal trajectory invariant `expect(2IySz)/expect(Ix) = g`.
- [ ] T074 Implement finite-time critical duration `Tcrit = arccot(2n)/(pi*J)`.
- [ ] T075 Implement finite-time `j(s)` function from Appendix B.
- [ ] T076 Implement equation solver for switching time `s` given total duration `T`.
- [ ] T077 Implement finite-time angles `h1`, `h2`.
- [ ] T078 Implement finite-time ROPE efficiency `g_T`.
- [ ] T079 Implement phase-I control `u1(t)` for finite-time ROPE.
- [ ] T080 Implement phase-II controls `u1(t)=u2(t)=1`.
- [ ] T081 Implement phase-III control symmetry `u2(t)=u1(T-t)`.
- [ ] T082 Implement RF amplitude conversion for phase-I and phase-III pulses.
- [ ] T083 Implement hard-pulse boundary flip-angle calculation.
- [ ] T084 Implement finite-time waveform sampler returning times, controls, amplitudes, and phases.
- [ ] T085 Add tests for `n=0` no-relaxation limit.
- [ ] T086 Add tests for paper example `n=1`, finite-time pulse values.
- [ ] T087 Add tests that finite-time efficiency approaches unconstrained `g` as `T` grows.
- [ ] T088 Add tests that `T <= Tcrit` reduces to INEPT controls.

### Phase 6: Analytical CROP And Decoherence Boundary Module

- [ ] T089 Implement `zeta = sqrt((ka^2-kc^2)/(J^2+kc^2))` with domain checks.
- [ ] T090 Implement CROP efficiency `eta = sqrt(1+zeta^2) - zeta`.
- [ ] T091 Implement primed CROP efficiency for spin S rates.
- [ ] T092 Implement physical limit for `Iz -> 2IzSz`.
- [ ] T093 Implement physical limit for `2IzSz -> Sz`.
- [ ] T094 Implement physical limit for `Iz -> Sz` as `eta*eta_prime`.
- [ ] T095 Implement physical limit for single-transition transfer `sqrt(eta^2 + eta_prime^2)`.
- [ ] T096 Implement limiting case `kc=0` reduction to ROPE-like expression.
- [ ] T097 Implement limiting case `kc/ka -> 1` approaching lossless transfer.
- [ ] T098 Implement single-transition decomposition helpers.
- [ ] T099 Implement slowly relaxing multiplet component identification.
- [ ] T100 Implement CROP pulse parameter container: amplitude, irradiation frequency, truncation window.
- [ ] T101 Implement truncated CROP waveform generator.
- [ ] T102 Implement CROP robustness sweep over `ka/J` and `kc/ka`.
- [ ] T103 Add tests for table of physical limits from the PNAS paper.
- [ ] T104 Add tests for monotonic improvement as `kc/ka` increases.
- [ ] T105 Add tests for no invalid efficiency above 1 except numerical tolerance.
- [ ] T106 Add regression tests for representative values `ka/J = 0.6`, `1.1`, `kc/ka = 0.75`.

### Phase 7: Propagation And GRAPE Cost Functions

- [ ] T107 Define `ControlProblem` dataclass matching Spinach `control` structure concepts.
- [ ] T108 Add required fields: `drifts`, `operators`, `rho_init`, `rho_targ`, `pulse_dt`, `pwr_levels`, `freeze`.
- [ ] T109 Add optional fields: fidelity mode, offsets, offset operators, phase cycle, basis, penalties, plotting, checkpoint.
- [ ] T110 Implement waveform shape validation for Cartesian XY controls.
- [ ] T111 Implement basis-coefficient expansion for smooth waveform bases.
- [ ] T112 Implement freeze-mask application to waveform updates and gradients.
- [ ] T113 Implement per-slice Hamiltonian/Liouvillian assembly.
- [ ] T114 Implement forward propagation cache over pulse slices.
- [ ] T115 Implement backward adjoint propagation cache.
- [ ] T116 Implement final-state evaluation for all source-target pairs.
- [ ] T117 Implement `grape_xy` returning fidelity only.
- [ ] T118 Implement first derivative of propagator by finite difference as a baseline.
- [ ] T119 Implement directional derivative of matrix exponential using auxiliary matrix method.
- [ ] T120 Replace baseline derivative in `grape_xy` with directional derivative.
- [ ] T121 Implement gradient accumulation over slices and control channels.
- [ ] T122 Implement Hessian placeholder returning `None` until full Hessian is ready.
- [ ] T123 Implement exact Hessian path for small dense systems.
- [ ] T124 Implement Liouville-space GRAPE variant.
- [ ] T125 Implement Hilbert-space GRAPE variant for closed systems.
- [ ] T126 Implement phase-only GRAPE adapter.
- [ ] T127 Implement amplitude-phase to XY conversion.
- [ ] T128 Implement curvilinear parameterization adapter.
- [ ] T129 Add gradient finite-difference tests for small two-spin systems.
- [ ] T130 Add Hessian finite-difference tests for tiny one-spin systems.
- [ ] T131 Add tests for `freeze` preserving frozen waveform entries.
- [ ] T132 Add tests for basis expansion preserving expected dimensions.

### Phase 8: Ensemble Control

- [ ] T133 Implement ensemble expansion over source-target pairs.
- [ ] T134 Implement ensemble expansion over multiple drift generators.
- [ ] T135 Implement ensemble expansion over control power levels.
- [ ] T136 Implement ensemble expansion over offsets.
- [ ] T137 Implement ensemble expansion over phase-cycle rows.
- [ ] T138 Implement Cartesian product ensemble builder.
- [ ] T139 Implement correlated ensemble mode `rho_match`.
- [ ] T140 Implement correlated ensemble mode `rho_drift`.
- [ ] T141 Implement ensemble fidelity aggregation.
- [ ] T142 Implement ensemble gradient aggregation.
- [ ] T143 Add optional multiprocessing or joblib backend abstraction.
- [ ] T144 Add deterministic serial backend for tests.
- [ ] T145 Add tests for ensemble size calculation.
- [ ] T146 Add tests for non-combinatorial correlated ensemble modes.
- [ ] T147 Add tests for offset operator inclusion in drift.
- [ ] T148 Add tests for phase-cycle transformation consistency.

### Phase 9: Penalties, Distortions, And Constraints

- [ ] T149 Implement norm-square penalty `NS`.
- [ ] T150 Implement spillout norm-square Cartesian penalty `SNS`.
- [ ] T151 Implement spillout norm-square amplitude penalty `SNSA`.
- [ ] T152 Implement derivative norm-square penalty `DNS`.
- [ ] T153 Implement penalty weights and separated reporting of simulation fidelity vs penalties.
- [ ] T154 Implement penalty gradients.
- [ ] T155 Implement penalty Hessians where practical.
- [ ] T156 Implement lower and upper waveform bounds.
- [ ] T157 Implement amplitude clipping helper for diagnostics only.
- [ ] T158 Implement no-op distortion model.
- [ ] T159 Implement tanh amplifier compression model.
- [ ] T160 Implement root amplifier compression model.
- [ ] T161 Implement single-pole filter distortion model.
- [ ] T162 Implement single-zero filter distortion model.
- [ ] T163 Implement transfer-matrix distortion model.
- [ ] T164 Implement distortion derivative hooks for GRAPE gradients.
- [ ] T165 Add tests for each penalty value and gradient.
- [ ] T166 Add tests for distortion identity and simple filter responses.

### Phase 10: Optimizers

- [ ] T167 Define optimizer result dataclass with final waveform, counts, convergence reason, and history.
- [ ] T168 Implement cubic/bracketed line search.
- [ ] T169 Implement gradient ascent baseline optimizer.
- [ ] T170 Implement LBFGS memory state.
- [ ] T171 Implement LBFGS inverse-Hessian update.
- [ ] T172 Implement LBFGS-GRAPE optimizer path.
- [ ] T173 Implement Newton-Raphson optimizer path for small waveform counts.
- [ ] T174 Implement Hessian regularization for Newton steps.
- [ ] T175 Implement RFO-style regularized Newton option if Hessian is indefinite.
- [ ] T176 Implement convergence checks for step norm `tol_x`.
- [ ] T177 Implement convergence checks for gradient norm `tol_g`.
- [ ] T178 Implement maximum iteration and maximum evaluation limits.
- [ ] T179 Implement optimizer checkpoint save.
- [ ] T180 Implement optimizer checkpoint resume.
- [ ] T181 Implement verbose iteration table matching the important Spinach diagnostics.
- [ ] T182 Add tests on quadratic toy objective.
- [ ] T183 Add tests that LBFGS improves GRAPE fidelity on a small state-transfer problem.
- [ ] T184 Add tests for checkpoint/resume equivalence.

### Phase 11: Input, Output, And Diagnostics

- [ ] T185 Implement waveform container with channels, units, time grid, metadata, and source problem hash.
- [ ] T186 Implement CSV waveform export.
- [ ] T187 Implement JSON waveform export.
- [ ] T188 Implement JSON waveform import.
- [ ] T189 Implement simple FAPT-like frequency-amplitude-phase-time import.
- [ ] T190 Implement Bruker pulse export stub with documented limitations.
- [ ] T191 Implement JCAMP-DX pulse import stub with documented limitations.
- [ ] T192 Implement heterodyne transformation helper.
- [ ] T193 Implement trajectory analysis for state populations.
- [ ] T194 Implement trajectory analysis for local spin expectation values.
- [ ] T195 Implement trajectory analysis for coherence order if basis metadata supports it.
- [ ] T196 Implement trajectory analysis for correlation order if basis metadata supports it.
- [ ] T197 Implement robustness histogram data generation.
- [ ] T198 Implement spectrogram data generation for channel pairs.
- [ ] T199 Add plotting functions for XY controls.
- [ ] T200 Add plotting functions for amplitude/phase controls.
- [ ] T201 Add plotting functions for ROPE/CROP efficiency curves.
- [ ] T202 Add plotting functions for state trajectories.
- [ ] T203 Add tests for waveform import/export round trips.
- [ ] T204 Add tests for plotting functions returning figure objects without display.

### Phase 12: Paper Reproduction Examples

- [ ] T205 Create example reproducing ROPE vs INEPT efficiency as a function of `n`.
- [ ] T206 Create example reproducing ROPE gain curve for `Ia -> 2IbSc`.
- [ ] T207 Create example reproducing in-phase transfer gain curve.
- [ ] T208 Create example plotting finite-time ROPE efficiency vs total transfer time.
- [ ] T209 Create example plotting finite-time ROPE controls for `n=1`.
- [ ] T210 Create example for sodium-formate-like parameters `J=193 Hz`, `T2=1.4 ms`.
- [ ] T211 Create example converting finite-time ROPE waveform to hard-pulse approximation.
- [ ] T212 Create example reproducing CROP efficiency vs `ka/J` for several `kc/ka`.
- [ ] T213 Create example comparing CROP with INEPT/CRIPT/CRINEPT placeholders.
- [ ] T214 Create example plotting CROP truncated amplitude and irradiation frequency.
- [ ] T215 Create example for CROP parameters `ka/J=0.6`, `1.1`, `kc/ka=0.75`.
- [ ] T216 Create example showing `kc/ka -> 1` decoherence-free limit.
- [ ] T217 Add expected-data snapshots for all paper reproduction examples.
- [ ] T218 Add tests that examples run in headless mode.

### Phase 13: Integration Milestones

- [ ] T219 Integrate operator core, state builders, and analytical ROPE into one public API.
- [ ] T220 Integrate CROP analytical limits with two-spin system factory.
- [ ] T221 Integrate `ControlProblem` validation with GRAPE cost functions.
- [ ] T222 Integrate penalties into `grape_xy` fidelity/gradient output.
- [ ] T223 Integrate ensemble expansion into `grape_xy`.
- [ ] T224 Integrate optimizer with `grape_xy` and checkpointing.
- [ ] T225 Integrate trajectory diagnostics with optimized waveforms.
- [ ] T226 Integrate waveform export with examples.
- [ ] T227 Add end-to-end test: analytical ROPE waveform beats INEPT for `n=1`.
- [ ] T228 Add end-to-end test: numerical GRAPE improves random guess for two-spin transfer.
- [ ] T229 Add end-to-end test: ensemble power robustness does not crash and reports histogram.
- [ ] T230 Add end-to-end test: optimizer resume reproduces uninterrupted result within tolerance.
- [ ] T231 Add end-to-end test: exported waveform can be imported and replayed.
- [ ] T232 Add documentation page showing how to define a spin system.
- [ ] T233 Add documentation page showing how to define initial and target states.
- [ ] T234 Add documentation page showing how to define drift and control operators.
- [ ] T235 Add documentation page showing how to run analytical ROPE/CROP calculations.
- [ ] T236 Add documentation page showing how to run numerical GRAPE.
- [ ] T237 Add documentation page showing how to use ensembles and penalties.
- [ ] T238 Add documentation page showing how to reproduce the paper figures.

### Phase 14: Validation, Quality, And Release Preparation

- [ ] T239 Add numerical tolerance policy for dense and sparse paths.
- [ ] T240 Add unit tests for all documented public functions.
- [ ] T241 Add regression tests for all analytical formulas.
- [ ] T242 Add finite-difference gradient test suite with seeded random problems.
- [ ] T243 Add performance benchmark for matrix exponential derivative methods.
- [ ] T244 Add performance benchmark for ensemble scaling.
- [ ] T245 Add ruff lint configuration and fix style issues.
- [ ] T246 Add mypy configuration for public dataclasses and APIs.
- [ ] T247 Add docstring coverage for public functions.
- [ ] T248 Add error-message tests for invalid dimensions and invalid physical parameters.
- [ ] T249 Add reproducibility controls for random initial guesses.
- [ ] T250 Add semantic versioning policy.
- [ ] T251 Add changelog.
- [ ] T252 Run full test suite and record baseline results.
- [ ] T253 Tag first internal milestone as analytical ROPE/CROP complete.
- [ ] T254 Tag second internal milestone as GRAPE dense complete.
- [ ] T255 Tag third internal milestone as ensemble/penalty complete.
- [ ] T256 Tag fourth internal milestone as examples/docs complete.

## Suggested Integration Order

1. Build operator/state foundations first (`T015-T062`).
2. Implement analytical ROPE and CROP next (`T063-T106`) because these provide exact validation targets.
3. Implement propagation and GRAPE gradients (`T107-T132`).
4. Add ensemble and penalty layers (`T133-T166`).
5. Add optimizers (`T167-T184`).
6. Add examples and diagnostics (`T185-T218`).
7. Run integration milestones and release-quality checks (`T219-T256`).

## Initial Acceptance Criteria

- Analytical ROPE functions reproduce the no-relaxation, finite-time, and high-relaxation limiting cases.
- Analytical CROP functions reproduce the `kc=0` and `kc/ka -> 1` limiting cases.
- A two-spin dense GRAPE example can optimize `Iz -> 2IzSz`.
- Gradient checks pass against finite differences on tiny systems.
- Example scripts reproduce the main qualitative curves from both PDFs.
- The API supports the major Spinach optimal-control concepts: drift/control operators, source-target pairs, ensemble dimensions, penalties, optimizer selection, checkpointing, and plotting diagnostics.
