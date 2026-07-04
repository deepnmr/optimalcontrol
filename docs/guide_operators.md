# Operators and Generators Guide

Operator builders create Hilbert-space matrices first, then convert them to
Liouville-space generators when needed. The two-spin Hilbert dimension is `4`;
the corresponding Liouville dimension is `16`.

## Drift and control operators

`drift_hamiltonian(sys)` returns the scalar-coupling Hamiltonian in rad/s.
`control_operators(sys)` returns dimensionless spin operators for the I and S RF
channels. Control amplitudes are supplied separately in rad/s when building the
total generator.

```python
import numpy as np

from optimalcontrol.spin_system import control_operators, drift_hamiltonian, two_spin_system

sys = two_spin_system(
    J_hz=140.0,
    kDD=0.0,
    kCSA_I=0.0,
    kCSA_S=0.0,
    ka=0.0,
    kc=0.0,
    ka_prime=0.0,
    kc_prime=0.0,
)

H_drift = drift_hamiltonian(sys)
controls = control_operators(sys)

assert H_drift.shape == (4, 4)
assert set(controls) == {"Ix", "Iy", "Sx", "Sy"}
assert controls["Ix"].shape == (4, 4)

print(f"drift Frobenius norm: {np.linalg.norm(H_drift):.3f} rad/s")
```

## Total Liouville generator

`total_generator(sys, controls)` assembles drift, chemical shifts, relaxation,
and RF controls into one Liouville-space generator. The `controls` mapping uses
the same names returned by `control_operators()`, with amplitudes in rad/s.

```python
import numpy as np
from scipy.linalg import expm

from optimalcontrol.operators import unvec, vec
from optimalcontrol.spin_system import total_generator, two_spin_system

sys = two_spin_system(
    J_hz=140.0,
    kDD=0.0,
    kCSA_I=0.0,
    kCSA_S=0.0,
    ka=0.0,
    kc=0.0,
    ka_prime=0.0,
    kc_prime=0.0,
)

rf_controls = {
    "Ix": 2.0 * np.pi * 1000.0,
    "Iy": 0.0,
    "Sx": 0.0,
    "Sy": 0.0,
}
generator = total_generator(sys, rf_controls)

rho0 = np.eye(4, dtype=np.complex128) / 4.0
rho1_vec = expm(generator * 1.0e-4) @ vec(rho0)
rho1 = unvec(rho1_vec, dim=4)

assert generator.shape == (16, 16)
assert rho1.shape == (4, 4)
```

Unknown control names raise `ValueError`, which helps catch mismatches between a
waveform channel list and the spin-system control operators.
