# GRAPE Guide

GRAPE code is organised around `ControlProblem`, a Spinach-style container for
drift generators, control generators, source and target states, pulse timing,
power levels, freeze masks, fidelity mode, ensemble metadata, penalties, and
checkpointing.

Waveforms inside GRAPE use shape `(n_steps, n_channels)`: rows are time slices
and columns are control channels.

## Build a ControlProblem

For Hilbert-space pure-state examples, provide anti-Hermitian generators such as
`-1j * Ix()` and pure-state vectors. Set `basis="hilbert"` so `grape_xy()` uses
the pure-state path.

```python
import numpy as np

from optimalcontrol.grape import ControlProblem, grape_xy
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.states import normalise_2norm

rho_init = np.array([1.0, 0.0], dtype=np.complex128)
rho_targ = normalise_2norm(
    np.array([0.35 + 0.15j, 0.88 - 0.28j], dtype=np.complex128)
)

cp = ControlProblem(
    drifts=[np.complex128(-1j) * 0.2 * Iz()],
    operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
    rho_init=[rho_init],
    rho_targ=[rho_targ],
    pulse_dt=0.05,
    pwr_levels=[0.8, 0.8],
    freeze=None,
    fidelity_mode="abs2",
    basis="hilbert",
)

wfm = np.zeros((16, 2), dtype=np.float64)
print(grape_xy(cp, wfm))
```

For density-matrix product-operator transfers, keep `basis="dense"` when the
drift and control matrices are Hilbert-space propagators. Use `basis="liouville"`
when the generators are Liouville-space superoperators and matrix states should
be vectorised before propagation.

## Fidelity and Gradient

`grape_xy()` returns the scalar fidelity. `grape_gradient()` returns the
analytical gradient with the same shape as the input waveform.

```python
import numpy as np

from optimalcontrol.grape import grape_gradient, grape_xy

wfm = np.array(
    [
        [0.02, 0.01],
        [0.03, -0.02],
        [0.01, 0.00],
    ],
    dtype=np.float64,
)

fidelity = grape_xy(cp, wfm)
gradient = grape_gradient(cp, wfm)

print(f"{fidelity:.6f}")
print(gradient.shape)
```

Frozen waveform entries are marked with a boolean mask shaped like the waveform.
Frozen entries are restored before propagation and their gradient entries are
zeroed.

```python
freeze = np.zeros((3, 2), dtype=np.bool_)
freeze[0, :] = True
cp.freeze = freeze

gradient = grape_gradient(cp, wfm)
assert np.all(gradient[0, :] == 0.0)
```

## Optimizer Selection

`run_grape()` is the top-level optimizer entry point. It accepts `method="lbfgs"`
or `method="newton"` and returns an exportable `Waveform` plus an `OptimResult`.

```python
import numpy as np

from optimalcontrol.optimizers import run_grape

wfm0 = np.zeros((16, 2), dtype=np.float64)
cp.freeze = None
waveform, result = run_grape(
    cp,
    wfm0,
    method="lbfgs",
    max_iter=25,
    tol_g=1e-8,
    produce_trajectory=True,
)

print(result.fidelity_final)
print(result.reason)
print(waveform.channels)
print(waveform.data.shape)  # exported layout is channels by time
```

Use L-BFGS for ordinary runs. Newton-Raphson uses the exact Hessian and is
available only for small waveforms, because `grape_hessian()` returns `None`
above 50 flattened parameters.

```python
small_wfm0 = np.zeros((4, 2), dtype=np.float64)
cp.freeze = None
_, newton_result = run_grape(
    cp,
    small_wfm0,
    method="newton",
    max_iter=10,
    regularise=True,
    rfo=False,
)

print(newton_result.n_iter)
```

## Checkpointing

Pass `checkpoint_path` either on the `ControlProblem` or in the optimizer call.
Optimizer checkpoints store the current waveform, fidelity history, evaluation
count, L-BFGS memory where relevant, and a signature binding the optimizer,
control problem, and waveform shape. A checkpoint cannot resume a different
method or problem. Legacy unsigned checkpoints are treated as waveform-only
warm starts; their history and optimizer memory are discarded.

```python
cp.checkpoint_path = "checkpoints/example-grape.json"
cp.freeze = None

_, partial = run_grape(cp, wfm0, method="lbfgs", max_iter=5)
_, resumed = run_grape(cp, wfm0, method="lbfgs", max_iter=25)

assert resumed.n_feval >= partial.n_feval
```

Function-level `checkpoint_path` overrides `cp.checkpoint_path` when both are
provided.
