# States Guide

State helpers in `optimalcontrol.states` build unnormalised product operators.
Normalise states explicitly before fidelity calculations.

## Product-operator labels

`state_from_label(label, n_spins)` supports one-spin labels on the first two
spins (`Ix`, `Iy`, `Iz`, `Sx`, `Sy`, `Sz`) and common two-spin product labels
with the conventional leading factor of 2 (`2IxSz`, `2IySz`, `2IzSx`,
`2IzSy`, `2IzSz`).

```python
from optimalcontrol.states import fidelity_real, normalise_hs, state_from_label

rho_init = normalise_hs(state_from_label("Iz", n_spins=2))
rho_target = normalise_hs(state_from_label("2IzSz", n_spins=2))

assert rho_init.shape == (4, 4)
assert rho_target.shape == (4, 4)
assert fidelity_real(rho_init, rho_init) == 1.0

print(f"initial-target direct overlap: {fidelity_real(rho_init, rho_target):.3f}")
```

The labels are product-operator states, not pulse instructions. For example,
`2IzSz` means `2 * Iz_0 * Sz_1` in Hilbert space.

## Single-transition operators

`single_transition_operator(label)` constructs two-spin single-transition
operators. Supported labels are `IzSalpha`, `IzSbeta`, `IxSbeta`, and
`IySbeta`. The `Salpha` and `Sbeta` suffixes select the alpha or beta projector
on the second spin.

```python
from optimalcontrol.states import normalise_hs, single_transition_operator

transition = single_transition_operator("IxSbeta")
transition_n = normalise_hs(transition)

assert transition.shape == (4, 4)
assert transition_n.shape == (4, 4)
```

## Normalisation policy

Use `normalise_hs()` for matrix product-operator states used by
`fidelity_real()`, `fidelity_abs2()`, and `fidelity_avg()`. Use
`normalise_2norm()` only when working directly with vectorised Liouville-space
states.

```python
from optimalcontrol.operators import vec
from optimalcontrol.states import normalise_2norm, normalise_hs, state_from_label

rho = state_from_label("Sx", n_spins=2)
rho_hs = normalise_hs(rho)
rho_vec = normalise_2norm(vec(rho))

assert rho_hs.shape == rho.shape
assert rho_vec.shape == (rho.size,)
```

Both normalisers raise `ValueError` for zero-norm inputs.
