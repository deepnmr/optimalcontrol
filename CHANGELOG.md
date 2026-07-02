# Changelog

## v0.3.0 - 2026-07-02

- Speed up the GRAPE hot path roughly 2x end to end. The Rust kernels now
  statically specialize the common generator dimensions (2, 3, 4, 8, 16) so
  small-matrix eigendecompositions and propagations run without heap
  allocation, validate anti-Hermiticity once per ensemble member instead of
  once per time slice, and parallelize over time slices when a problem has
  fewer members than worker threads.
- Feed ensemble problems to the native kernels directly from the unexpanded
  `ControlProblem` (numpy broadcasting over the drift/power/offset axes),
  replacing the per-member `ControlProblem` materialization that previously
  dominated Python-side overhead in optimizer loops.
- Keep a general dense-`expm` fallback in the fidelity kernel for
  non-anti-Hermitian (dissipative) generators, and the pure NumPy/SciPy path
  via `OPTIMALCONTROL_DISABLE_RUST=1`.
- Ship the minimum-power and minimum-length REBURP-style methyl 180 example
  modules with their regression tests, cached waveforms, and Bruker shape
  exports (described under v0.2.0 but not previously committed).

## v0.2.0 - 2026-06-23

- Add a PyO3/Rust accelerator for coherent GRAPE fidelity and exact gradients.
- Parallelize offset and RF-power ensemble propagation with Rayon.
- Add native Bloch profile propagation used by the broadband pulse examples.
- Retain an opt-out NumPy/SciPy fallback via `OPTIMALCONTROL_DISABLE_RUST=1`.
- Add reproducible broadband GRAPE, ReBURP, phase-only inversion, methyl/water,
  HMQC artifact-suppression, and INEPT pulse-design examples.
- Add a minimum max-power REBURP-style methyl 180 example that maps the
  (peak-RF, duration) Pareto frontier and caches the 6.0 kHz / 2.60 ms knee
  (40 percent lower peak field than the 10 kHz siblings).
- Add the minimum-length companion at the opposite frontier end: 10 kHz /
  1.80 ms, the shortest passing smooth methyl 180.
- Thread an explicit `rf_max_hz` through the methyl/water evaluation and profile
  helpers so peak-field sweeps reuse the dense validator.
- Ship cached regression waveforms and Bruker shape exports for the new examples.

All notable changes to this project are recorded here.

## v0.1.0 - 2026-04-23

### Internal Milestones

- Analytical ROPE/CROP complete.
- GRAPE dense complete.
- Ensemble/penalty complete.
- Examples/docs complete.

### Quality Baseline

- `python3 -m pytest`: 154 passed on 2026-04-23.
