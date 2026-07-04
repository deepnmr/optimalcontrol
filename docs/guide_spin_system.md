# Spin System Guide

Spin systems are described with small dataclasses in `optimalcontrol.spin_system`.
The package uses the convention that spin index `0` is the source/I spin and spin
index `1` is the target/S spin for the two-spin paper models.

Public coupling constants and chemical shifts are in Hz. Relaxation fields on
`RelaxationRates` are stored in rad/s.

## Two-spin factory

Use `two_spin_system()` for the on-resonance heteronuclear I-S model used by the
ROPE and CROP examples. It creates two spins, one scalar coupling, zero chemical
shifts, and a `RelaxationRates` object.

```python
import numpy as np

from optimalcontrol.spin_system import two_spin_system

J_hz = 140.0
sys = two_spin_system(
    J_hz=J_hz,
    kDD=2.0 * np.pi * 2.0,
    kCSA_I=2.0 * np.pi * 1.0,
    kCSA_S=2.0 * np.pi * 1.5,
    ka=0.0,
    kc=0.0,
    ka_prime=0.0,
    kc_prime=0.0,
)

assert [spin.channel for spin in sys.spins] == ["I", "S"]
assert sys.couplings[0].J_hz == J_hz
assert sys.shifts_hz == {0: 0.0, 1: 0.0}

print(f"I-spin relaxation rate: {sys.relaxation.kI:.3f} rad/s")
```

## Explicit SpinSystem construction

Construct `SpinSystem` directly when the model needs custom isotopes, shifts, or
couplings. The Hamiltonian builders consume the same dataclasses used by the
factory, so direct construction and factory construction are interchangeable once
the fields are populated.

```python
import numpy as np

from optimalcontrol.spin_system import Coupling, RelaxationRates, Spin, SpinSystem

custom = SpinSystem(
    spins=[
        Spin(isotope="1H", gamma=267.52218744e6, channel="I"),
        Spin(isotope="13C", gamma=67.2828402e6, channel="S"),
    ],
    couplings=[Coupling(spin_i=0, spin_j=1, J_hz=193.0)],
    shifts_hz={0: 0.0, 1: 12.5},
    relaxation=RelaxationRates(
        kDD=2.0 * np.pi * 3.0,
        kCSA_I=2.0 * np.pi * 1.0,
        kCSA_S=2.0 * np.pi * 1.2,
        ka=0.0,
        kc=0.0,
        ka_prime=0.0,
        kc_prime=0.0,
    ),
    basis="dense",
)

assert custom.spins[0].isotope == "1H"
assert custom.couplings[0].spin_i == 0
assert custom.shifts_hz[1] == 12.5
```

Keep the spin indices consistent across `spins`, `couplings`, and `shifts_hz`.
Builders validate that each referenced spin index exists and raise `ValueError`
for invalid couplings or unsupported non-two-spin control/relaxation helpers.
