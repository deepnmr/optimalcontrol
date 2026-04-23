# Equations and Notation Conventions

## Unit Policy

**Rule:** Public API accepts and returns frequencies in **Hz**. Internal calculations use **rad/s**.

Conversions occur at clearly identified boundaries:

| Boundary | Direction | Conversion |
|---|---|---|
| `drift_hamiltonian(sys)` input | Hz → rad/s | multiply `J_hz` by `2π` |
| `shift_hamiltonian(sys)` input | Hz → rad/s | multiply `shifts_hz` values by `2π` |
| `rope_n(kI, J)` inputs | Hz (both) | dimensionless ratio, no conversion needed |
| `rope_Tcrit(n, J_hz)` | Hz → rad/s | multiply `J_hz` by `2π` internally |
| `crop_zeta(ka, kc, J_hz)` | Hz (all three) | dimensionless ratio, no conversion needed |
| `inept_efficiency(t, J_hz, k_hz)` | Hz → rad/s | multiply by `π` where `pi*J*t` appears |
| `RelaxationRates` fields | stored in rad/s | document units in field docstring |

Wherever a function returns a time (in seconds), it is consistent with Hz inputs.

---

## State Normalisation Policy

**Rule:** Hilbert-Schmidt norm is used at **all fidelity evaluation points**.

Definition:
```
||ρ||_HS = sqrt(Tr(ρ† ρ))
```

Normalised state: `ρ_n = ρ / ||ρ||_HS`

All `state_from_label` and `single_transition_operator` results are **not pre-normalised** by
default; callers must apply `normalise_hs(v)` before computing fidelities. The `fidelity_real`
and `fidelity_abs2` functions expect pre-normalised inputs; they do **not** normalise internally.

The 2-norm (`normalise_2norm`) is provided for use with vectorised Liouville-space states where
the vector `vec(ρ)` is used directly. Hilbert-Schmidt norm on a matrix equals the 2-norm of its
vectorised form:
```
||ρ||_HS = ||vec(ρ)||_2
```

---

## Numerical Tolerance Policy

Release checks use these tolerances unless a test documents a stricter analytical regression.

- **Dense path:** `rtol = 1e-10` for gradient and propagator checks.
- **Sparse path:** `rtol = 1e-8` for gradient and propagator checks.
- **Analytical formula tests:** `rtol = 1e-8` unless a tighter tolerance is specified per story.
- **Finite-difference gradient checks:** `rtol = 1e-5` (limited by step-size accuracy).

---

## ROPE Symbol Table (JMR 2003)

| Symbol | Definition | Units | Python identifier |
|---|---|---|---|
| I, S | Spin-1/2 nuclei (I = source, S = target) | — | — |
| J | Scalar coupling constant | Hz | `J_hz` |
| kI | Transverse relaxation rate of I | rad/s | `kI` (= `kDD + kCSA_I`) |
| kS | Transverse relaxation rate of S | rad/s | `kS` |
| n | Relative relaxation `n = kI / J` | dimensionless | `n` |
| g(n) | ROPE efficiency `sqrt(1+n²) − n` | dimensionless | `rope_g(n)` |
| t* | INEPT optimal time `arccot(n) / (πJ)` | s | `inept_optimal_time(n, J_hz)` |
| V(r1,r2) | Optimal return function `sqrt(g²r1² + r2²)` | same as r1,r2 | `rope_V(r1, r2, n)` |
| u1, u2 | Control amplitudes on I and S channels | rad/s | `u1`, `u2` |
| u2/u1 | Optimal control ratio `g r1 / r2` | dimensionless | `rope_u_ratio(r1, r2, n)` |
| r1 | `<Ix>` expectation value | dimensionless | `r1` |
| r2 | `<2IySz>` expectation value | dimensionless | `r2` |
| T | Total transfer duration (finite-time) | s | `T` |
| T_crit | Critical duration `arccot(2n) / (πJ)` | s | `rope_Tcrit(n, J_hz)` |
| s | Switching time (phase I → II boundary) | s | `s` |
| j(s) | Appendix B transcendental function | dimensionless | `rope_j(s, n)` |
| h1, h2 | Finite-time phase angles | rad | `h1`, `h2` |
| g_T | Finite-time ROPE efficiency | dimensionless | `rope_finite_efficiency(T, n, J_hz)` |
| g_in | In-phase transfer efficiency `Ix → Sx` | dimensionless | `rope_g_inphase(n)` |

### Phase structure of finite-time ROPE

```
Phase I   : t ∈ [0, s]        — bang arc, u1(t) and u2(t) from formulae
Phase II  : t ∈ [s, T-s]      — singular arc, u1 = u2 = 1 exactly
Phase III : t ∈ [T-s, T]      — bang arc, u2(t) = u1(T-t) by symmetry
```

---

## CROP Symbol Table (PNAS 2003)

| Symbol | Definition | Units | Python identifier |
|---|---|---|---|
| ka | DD-CSA cross-corr rate (I longitudinal) | rad/s | `ka` |
| kc | DD-CSA cross-corr rate (I transverse) | rad/s | `kc` |
| ka′ | DD-CSA cross-corr rate (S longitudinal) | rad/s | `ka_prime` |
| kc′ | DD-CSA cross-corr rate (S transverse) | rad/s | `kc_prime` |
| ζ | `sqrt((ka²−kc²) / (J²+kc²))` | dimensionless | `crop_zeta(ka, kc, J_hz)` |
| η | CROP efficiency `sqrt(1+ζ²) − ζ` | dimensionless | `crop_eta(ka, kc, J_hz)` |
| η′ | CROP efficiency for S rates | dimensionless | `crop_eta_prime(ka_prime, kc_prime, J_hz)` |

### Physical transfer limits

| Transfer | Efficiency | Python function |
|---|---|---|
| Iz → 2IzSz | η | `crop_limit_Iz_to_2IzSz(eta)` |
| 2IzSz → Sz | η′ | `crop_limit_2IzSz_to_Sz(eta_prime)` |
| Iz → Sz | η η′ | `crop_limit_Iz_to_Sz(eta, eta_prime)` |
| Single transition | sqrt(η² + η′²) | `crop_limit_single_transition(eta, eta_prime)` |

### Limiting cases

- **kc = 0:** `η(ka, 0, J) = g(n=ka/J)` — reduces to ROPE efficiency
- **kc → ka:** `η → 1` — approaches lossless (decoherence-free) transfer
- **ka = kc = 0:** `η = 1` — no cross-correlation, lossless limit

---

## Liouville-Space Conventions

Vectorisation uses **column-major (Fortran) stacking** unless otherwise specified in the
function docstring:
```
vec(ρ) = ρ.flatten(order='F')
```

Left multiplication superoperator: `L(A) = I ⊗ A`
Right multiplication superoperator: `R(A) = A^T ⊗ I`

Liouvillian commutator: `L_comm(A) = −i (L(A) − R(A))`

The vectorisation convention is documented in `vec()` and `unvec()` docstrings.
Any deviation must be documented explicitly.

---

## Operator Algebra (Hilbert Space)

Spin-1/2 operators (dimensionless, in units of ℏ):

```
Ix = [[0, 1/2], [1/2, 0]]
Iy = [[0, -i/2], [i/2, 0]]
Iz = [[1/2, 0], [0, -1/2]]
Ip = Ix + i*Iy = [[0, 1], [0, 0]]
Im = Ix - i*Iy = [[0, 0], [1, 0]]
E  = [[1, 0], [0, 1]]
```

Commutation relations:
```
[Ix, Iy] = i Iz   (cyclic: [Iy, Iz] = i Ix, [Iz, Ix] = i Iy)
```

Multi-spin operators use the Kronecker product. For a two-spin system:
```
Iz ⊗ I  (spin 0 only)   = place_operator(Iz(), 0, 2)
I  ⊗ Sz (spin 1 only)   = place_operator(Iz(), 1, 2)
2IzSz                   = 2 * kron(Iz(), Iz())
```
