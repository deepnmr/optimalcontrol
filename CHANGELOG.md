# Changelog

## v0.2.0 - 2026-06-23

- Add a PyO3/Rust accelerator for coherent GRAPE fidelity and exact gradients.
- Parallelize offset and RF-power ensemble propagation with Rayon.
- Add native Bloch profile propagation used by the broadband pulse examples.
- Retain an opt-out NumPy/SciPy fallback via `OPTIMALCONTROL_DISABLE_RUST=1`.
- Add reproducible broadband GRAPE, ReBURP, phase-only inversion, methyl/water,
  HMQC artifact-suppression, and INEPT pulse-design examples.
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
