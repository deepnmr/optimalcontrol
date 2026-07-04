# Spinach → Python Module Mapping

This document maps Spinach MATLAB concepts to their Python equivalents in the `optimalcontrol`
package.

## Control Problem Structure

| Spinach `control` field | Python `ControlProblem` field | Notes |
|---|---|---|
| `drifts` | `drifts: list[ndarray]` | List of drift Liouvillians/Hamiltonians |
| `operators` | `operators: list[ndarray]` | Control operators in same space as drifts |
| `rho_init` | `rho_init: list[ndarray]` | Initial density matrices (vectorised if Liouville) |
| `rho_targ` | `rho_targ: list[ndarray]` | Target density matrices |
| `pulse_dt` | `pulse_dt: float` | Time step per slice (seconds) |
| `pwr_levels` | `pwr_levels: list[float]` | RF power scaling factors for ensemble |
| `freeze` | `freeze: ndarray \| None` | Boolean mask; True = frozen (not updated) |
| `fidelity_type` | `fidelity_mode: str` | `'real'`, `'abs2'` |
| `offset_op` | `offset_operators: list[ndarray] \| None` | Operator multiplied by each offset value |
| `offsets` | `offsets: list[float] \| None` | Offset frequencies in Hz |
| `phase_cycle` | `phase_cycle: ndarray \| None` | Phase rotation matrix per cycle step |
| `basis` | `basis: str` | `'dense'` or `'sparse'` |
| `penalties` | `penalties: list \| None` | Active penalty function descriptors |
| `checkpoint` | `checkpoint_path: str \| None` | Path for save/resume |

## GRAPE Functions

| Spinach function | Python function | Module |
|---|---|---|
| `grape_xy(problem, guess)` | `grape_xy(cp, wfm)` | `optimalcontrol.grape` |
| `grape_xy` fidelity return | `grape_xy` → `float` | Single scalar, same convention |
| `grape_xy` gradient | `grape_gradient(cp, wfm)` | Separate function |
| `dirdiff(A, dA, dt)` | `dir_diff_expm(H, dH, dt)` | `optimalcontrol.grape` |
| `fminnewton` | `newton_raphson(cp, wfm0, ...)` | `optimalcontrol.optimizers` |
| `fminnewton` LBFGS path | `lbfgs_grape(cp, wfm0, ...)` | `optimalcontrol.optimizers` |
| top-level entry | `run_grape(cp, wfm0, method=...)` | `optimalcontrol.optimizers` |

## Operator Construction

| Spinach concept | Python function | Module |
|---|---|---|
| `operator(sys,'Ix','1H')` | `Ix()`, `place_operator(Ix(), 0, n)` | `optimalcontrol.operators` |
| Kronecker product | `kron_product([A, B, ...])` | `optimalcontrol.operators` |
| `2IzSz` product | `two_spin_product(Iz(), Iz())` | `optimalcontrol.operators` |
| Commutator `[A,B]` | `comm(A, B)` | `optimalcontrol.operators` |
| Liouville vectorise | `vec(rho)` | `optimalcontrol.operators` |
| Liouville un-vectorise | `unvec(v, dim)` | `optimalcontrol.operators` |
| Left multiplication superop | `L_op(A)` | `optimalcontrol.operators` |
| Right multiplication superop | `R_op(A)` | `optimalcontrol.operators` |
| Liouvillian commutator | `liouvillian_comm(A)` | `optimalcontrol.operators` |
| Lindblad dissipator | `lindblad_dissipator(Fk_list, a_kl)` | `optimalcontrol.operators` |

## Spin System

| Spinach concept | Python equivalent | Module |
|---|---|---|
| `sys.isotopes` cell array | `SpinSystem.spins: list[Spin]` | `optimalcontrol.spin_system` |
| `inter.coupling.scalar` | `SpinSystem.couplings: list[Coupling]` | `optimalcontrol.spin_system` |
| `inter.relaxation` rates | `SpinSystem.relaxation: RelaxationRates` | `optimalcontrol.spin_system` |
| Chemical shift `inter.zeeman.scalar` | `SpinSystem.shifts_hz: dict[int, float]` | `optimalcontrol.spin_system` |
| Two-spin factory | `two_spin_system(J_hz, kDD, ...)` | `optimalcontrol.spin_system` |
| Drift Liouvillian | `drift_hamiltonian(sys)` | `optimalcontrol.spin_system` |
| Control operators | `control_operators(sys)` | `optimalcontrol.spin_system` |
| Relaxation Liouvillian | `relaxation_liouvillian(sys)` | `optimalcontrol.spin_system` |
| Cross-correlated relaxation | `relaxation_liouvillian_crosscorr(sys)` | `optimalcontrol.spin_system` |
| Total generator | `total_generator(sys, controls)` | `optimalcontrol.spin_system` |

## States and Fidelity

| Spinach concept | Python function | Module |
|---|---|---|
| `state(sys, 'Iz', '1H')` | `state_from_label('Iz', n_spins)` | `optimalcontrol.states` |
| Single-transition operator | `single_transition_operator(label)` | `optimalcontrol.states` |
| HS normalisation | `normalise_hs(v)` | `optimalcontrol.states` |
| 2-norm normalisation | `normalise_2norm(v)` | `optimalcontrol.states` |
| `hdot` overlap | `fidelity_real(rho_f, rho_t)` | `optimalcontrol.states` |
| Absolute-square fidelity | `fidelity_abs2(rho_f, rho_t)` | `optimalcontrol.states` |
| Averaged fidelity | `fidelity_avg(rho_f_list, rho_t_list)` | `optimalcontrol.states` |

## Waveform I/O

| Spinach concept | Python function | Module |
|---|---|---|
| Waveform struct | `Waveform` dataclass | `optimalcontrol.io` |
| `grape_save` / `grape_load` | `export_json` / `import_json` | `optimalcontrol.io` |
| Bruker export | `export_bruker` (stub) | `optimalcontrol.io` |
| Heterodyne shift | `heterodyne_transform` | `optimalcontrol.io` |

## Ensemble Expansion

| Spinach ensemble dimension | Python function | Module |
|---|---|---|
| Multiple drifts | `expand_drifts(cp)` | `optimalcontrol.ensemble` |
| Power levels | `expand_power_levels(cp)` | `optimalcontrol.ensemble` |
| Offset frequencies | `expand_offsets(cp)` | `optimalcontrol.ensemble` |
| Phase cycle | `expand_phase_cycle(cp)` | `optimalcontrol.ensemble` |
| Cartesian product | `cartesian_product_ensemble(cp)` | `optimalcontrol.ensemble` |
| Correlated `rho_match` | `correlated_rho_match(cp)` | `optimalcontrol.ensemble` |
| Correlated `rho_drift` | `correlated_rho_drift(cp)` | `optimalcontrol.ensemble` |

## Penalties

| Spinach penalty | Python function | Module |
|---|---|---|
| `NS` norm-square | `penalty_NS(wfm, weight)` | `optimalcontrol.penalties` |
| `SNS` spillout Cartesian | `penalty_SNS(wfm, limit, weight)` | `optimalcontrol.penalties` |
| `SNSA` spillout amplitude | `penalty_SNSA(wfm, limit, weight)` | `optimalcontrol.penalties` |
| `DNS` derivative norm-square | `penalty_DNS(wfm, weight)` | `optimalcontrol.penalties` |
| Combined penalty | `total_penalty(wfm, penalty_list)` | `optimalcontrol.penalties` |
