# Ensembles and Penalties Guide

Ensemble helpers expand one `ControlProblem` into concrete GRAPE problems and
average fidelity or gradients over the expanded members. Penalties are
minimisation-style waveform costs that are subtracted from the GRAPE fidelity
objective.

## Ensemble Dimensions

The Cartesian ensemble path supports four independent axes:

| Axis | `ControlProblem` field | Expansion helper |
|---|---|---|
| Drift generators | `drifts` with more than one matrix | `expand_drifts()` |
| RF power levels | `pwr_levels` length differs from operator count | `expand_power_levels()` |
| Offsets | `offsets` and `offset_operators` | `expand_offsets()` |
| Phase cycle | `phase_cycle` | `expand_phase_cycle()` |

When `len(pwr_levels) == len(operators)`, the values are ordinary per-channel
control scaling factors. They become an RF-power ensemble axis only when the
length differs from the number of control operators.

```python
import numpy as np

from optimalcontrol.ensemble import cartesian_product_ensemble, ensemble_fidelity
from optimalcontrol.grape import ControlProblem
from optimalcontrol.operators import Ix, Iz
from optimalcontrol.states import normalise_2norm

rho_init = np.array([1.0, 0.0], dtype=np.complex128)
rho_targ = normalise_2norm(
    np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128)
)

cp = ControlProblem(
    drifts=[
        np.complex128(-1j) * 0.2 * Iz(),
        np.complex128(-1j) * -0.1 * Iz(),
    ],
    operators=[np.complex128(-1j) * Ix()],
    rho_init=[rho_init],
    rho_targ=[rho_targ],
    pulse_dt=0.05,
    pwr_levels=[0.8, 1.2],
    freeze=None,
    fidelity_mode="abs2",
    basis="hilbert",
)

wfm = np.array([[0.12], [-0.04], [0.08]], dtype=np.float64)
members = cartesian_product_ensemble(cp)
fidelity = ensemble_fidelity(cp, wfm)

print(len(members))  # two drifts times two RF scales
print(f"{fidelity:.6f}")
```

`grape_xy()` dispatches to the ensemble path automatically when ensemble axes
are active, so explicit calls to `ensemble_fidelity()` are mainly useful for
inspection and tests.

## Offsets and Phase Cycles

Offsets add `offset * sum(offset_operators)` to each drift generator. Phase-cycle
rows are radians and rotate the initial states, either with one global phase per
row or one phase per source state.

```python
cp.offsets = [-0.25, 0.0, 0.25]
cp.offset_operators = [np.complex128(-1j) * Iz()]
cp.phase_cycle = np.array([0.0, np.pi], dtype=np.float64)

members = cartesian_product_ensemble(cp)
print(len(members))  # drifts * RF scales * offsets * phase rows
```

Expanded members clear consumed metadata such as `offsets`, `offset_operators`,
and `phase_cycle`, making each expanded problem compatible with the single-member
GRAPE propagation path.

## Correlated Modes

`correlated_rho_match()` splits matched source-target pairs into separate
single-pair problems. `correlated_rho_drift()` also matches each drift by index
and requires one drift per source-target pair.

```python
from optimalcontrol.ensemble import correlated_rho_drift, correlated_rho_match

matched = correlated_rho_match(cp)
print(len(matched))

cp.drifts = cp.drifts[: len(cp.rho_init)]
rho_drift = correlated_rho_drift(cp)
print(len(rho_drift))
```

Use correlated modes when the ensemble members are coupled by construction
rather than forming a Cartesian product.

## Penalty Configuration

Penalties are configured with `PenaltySpec` entries or custom callables that
return `(value, gradient)`. Built-in kinds are:

| Kind | Meaning | Required fields |
|---|---|---|
| `NS` | elementwise norm-square | `weight` |
| `SNS` | elementwise Cartesian spillout | `weight`, `limit` |
| `SNSA` | row-amplitude spillout across channels | `weight`, `limit` |
| `DNS` | adjacent-row finite-difference norm-square | `weight` |

```python
import numpy as np

from optimalcontrol.grape import grape_gradient, grape_xy
from optimalcontrol.penalties import PenaltySpec, total_penalty

cp.penalties = [
    PenaltySpec("NS", weight=1.0e-4),
    PenaltySpec("SNSA", weight=1.0e-2, limit=0.2),
    PenaltySpec("DNS", weight=1.0e-3),
]

wfm = np.array(
    [
        [0.10],
        [0.25],
        [0.20],
    ],
    dtype=np.float64,
)

penalty_value, penalty_gradient = total_penalty(wfm, cp.penalties)
score = grape_xy(cp, wfm)
gradient = grape_gradient(cp, wfm)

print(penalty_value)
print(penalty_gradient.shape)
print(score)
print(gradient.shape)
```

Because penalties are subtracted from the fidelity, their gradients are also
subtracted from the GRAPE fidelity gradient. Freeze masks are applied after
penalties so frozen entries remain zero-gradient.
