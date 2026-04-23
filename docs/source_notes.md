# Source Notes

## JMR 2003 — ROPE (Relaxation-Optimised Pulse Engineering)

**Reference:** Khaneja et al., "Optimal control of spin dynamics in the presence of relaxation",
*Journal of Magnetic Resonance* 162 (2003) 311–319.
**File:** `refdoc/1-s2.0-S109078070300003X-main.pdf`

### What this paper contributes

- **Physical model:** On-resonance two-spin I-S heteronuclear system under secular relaxation with
  rates `kI = kDD + kCSA_I` and `kS = kDD + kCSA_S`. No DD-CSA cross-correlation in the main text;
  cross-correlated case is covered by the PNAS 2003 paper.

- **Unconstrained ROPE efficiency (Section 2):**
  - Relative relaxation parameter `n = kI / J`
  - Optimal coherence-transfer efficiency `g(n) = sqrt(1 + n²) − n`
  - Comparison with INEPT maximum efficiency and the ROPE gain factor

- **Optimal return function (Section 2):**
  - `V(r1, r2) = sqrt(g² r1² + r2²)` — Lyapunov function on the Bloch-ball state space
  - Unconstrained optimal control ratio `u2/u1 = g r1 / r2`
  - Trajectory invariant: `<2IySz> / <Ix> = g` along the optimal path

- **Finite-time ROPE (Section 3 + Appendix B):**
  - Critical duration `T_crit = arccot(2n) / (π J)` below which free-evolution (INEPT-like) is optimal
  - Three-phase bang–singular–bang structure: phase I (0 to s), phase II (s to T-s), phase III (T-s to T)
  - Switching time `s` solved numerically from the `j(s)` equation in Appendix B
  - Finite-time efficiency `g_T`, flip angles `h1` and `h2`
  - Phase-I control formulae for `u1(t)` and `u2(t)` as functions of the instantaneous state

- **Waveform generation:**
  - Phase-II region has `u1 = u2 = 1` exactly
  - Phase-III uses symmetry `u2(t) = u1(T − t)`
  - Hard-pulse approximation converts continuous controls to flip angles in degrees

- **Transfer types addressed:**
  - Antiphase transfer `Iz → 2IzSz` (main text)
  - In-phase transfer `Ix → Sx` (Section 4, `g_in` formula)
  - `Iz → 2IbSc` gain curve (Figure 4)

---

## PNAS 2003 — CROP (Cross-Relaxation Optimised Pulses)

**Reference:** Khaneja, Luy, Glaser, "Boundary of quantum evolution under decoherence",
*PNAS* 100 (2003) 13162–13166.
**File:** `refdoc/khaneja-et-al-2003-boundary-of-quantum-evolution-under-decoherence.pdf`

### What this paper contributes

- **Physical model:** Two-spin I-S system including DD-CSA cross-correlated relaxation. Introduces
  four additional rates `ka`, `kc`, `ka′`, `kc′` that couple longitudinal and transverse coherences
  via the correlated fluctuation of DD and CSA interactions.

- **CROP efficiency (main result):**
  - `ζ = sqrt((ka² − kc²) / (J² + kc²))`
  - `η = sqrt(1 + ζ²) − ζ` — transfer efficiency for `Iz → 2IzSz`
  - Primed version `η′` for `2IzSz → Sz` using `ka′`, `kc′`

- **Physical transfer limits:**
  - `Iz → 2IzSz`: efficiency `η`
  - `2IzSz → Sz`: efficiency `η′`
  - `Iz → Sz` (two-step): `η η′`
  - Single-transition transfer: `sqrt(η² + η′²)`

- **Limiting cases:**
  - `kc = 0` recovers ROPE: `η(kc=0) = g(n=ka/J)`
  - `kc/ka → 1` approaches lossless transfer `η → 1`

- **Truncated CROP waveform:**
  - Pulse amplitude and irradiation frequency parameterised by `ka`, `kc`, `J`
  - Truncation window parameter controls pulse duration

- **Robustness:**
  - 2D sweep over `(ka/J, kc/ka)` gives decoherence boundary contour plots

---

## Spinach Optimal Control Module

**Reference:** https://spindynamics.org/wiki/index.php?title=Optimal_control_module
**Key MATLAB files:** `grape_xy.m`, `fminnewton.m`, `dirdiff.m`

### What Spinach contributes to the implementation

- **`ControlProblem` concept:** Spinach's `control` struct provides the template for the
  `ControlProblem` dataclass: drift operators, control operators, initial/target states,
  pulse time step, power levels, freeze mask, fidelity mode, offsets, and penalties.

- **GRAPE cost function:** `grape_xy.m` is the reference for the forward/backward propagation
  loop, fidelity accumulation over source-target pairs, and the return convention
  (fidelity scalar only; gradient requested separately).

- **Directional derivative:** `dirdiff.m` implements the auxiliary 2×2 block matrix method for
  computing `d/dε expm(−i(H + ε dH) dt)|_{ε=0}`, which is the analytical gradient of each
  propagator slice.

- **Optimizer:** `fminnewton.m` provides the Newton-Raphson convergence diagnostics format
  (iteration table with fidelity, penalty, gradient norm, step norm) that the Python
  `print_iteration_table` function should match.

- **Ensemble dimensions:** Spinach supports drifts, power levels, offsets, and phase cycles as
  independent ensemble axes combined by Cartesian product, plus correlated modes `rho_match` and
  `rho_drift`. These map directly to the `expand_*` functions in `optimalcontrol/ensemble.py`.
