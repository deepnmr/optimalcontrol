# ROPE and CROP Guide

Analytical ROPE and CROP helpers live in `optimalcontrol.rope` and
`optimalcontrol.crop`. The paper-level APIs use Hz-style frequency scalars for
dimensionless ratios, and waveform samplers return physical times in seconds.

## ROPE Efficiency

`rope_g(n)` returns the unconstrained relaxation-optimised transfer efficiency
for the relative relaxation rate `n = kI / J`. `inept_max_efficiency()` gives the
best matching INEPT value for comparison.

```python
import numpy as np

from optimalcontrol.rope import inept_max_efficiency, rope_g, rope_n

J_hz = 100.0
kI_hz = 80.0
n = rope_n(kI_hz, J_hz)

g_rope = rope_g(n)
g_inept = inept_max_efficiency(n, J_hz)

print(f"n = {n:.3f}")
print(f"ROPE efficiency = {g_rope:.3f}")
print(f"INEPT max = {g_inept:.3f}")
print(f"gain = {g_rope / g_inept:.3f}")
```

For finite transfer durations, `rope_waveform()` samples the three-phase
bang-singular-bang control law. The returned dictionary contains `times`, `u1`,
`u2`, `amplitude`, and `phase`.

```python
import numpy as np

from optimalcontrol.rope import rope_Tcrit, rope_waveform

J_hz = 100.0
n = 1.0
T = 3.0 / J_hz
dt = 1.0e-4

assert T > rope_Tcrit(n, J_hz)
wfm = rope_waveform(T=T, n=n, J_hz=J_hz, dt=dt)

print(wfm["times"].shape)
print(np.nanmax(wfm["amplitude"]))
```

`amplitude` is an angular RF amplitude in rad/s. `phase` is in radians: phase I
uses `pi/2`, the central singular arc is zero-amplitude, and phase III uses `0`.

## Sodium Formate Worked Example

The JMR sodium-formate example uses `J = 193 Hz` and `T2 = 1.4 ms`. This package
uses `k_hz = 1 / (pi * T2)` so that the INEPT decay factor
`exp(-pi * k_hz * t)` equals `exp(-t / T2)`.

```python
import math

from optimalcontrol.rope import (
    inept_max_efficiency,
    rope_Tcrit,
    rope_g,
    rope_n,
    rope_waveform,
)

J_hz = 193.0
T2 = 1.4e-3
k_hz = 1.0 / (math.pi * T2)
n = rope_n(k_hz, J_hz)

g_inf = rope_g(n)
inept_best = inept_max_efficiency(n, J_hz)
Tcrit = rope_Tcrit(n, J_hz)

print(f"n = {n:.3f}")
print(f"ROPE g(infinity) = {g_inf:.3f}")
print(f"INEPT max = {inept_best:.3f}")
print(f"Tcrit = {Tcrit * 1e3:.3f} ms")

waveform = rope_waveform(T=3.0e-3, n=n, J_hz=J_hz, dt=2.0e-5)
print(f"{len(waveform['times'])} slices")
```

The same calculation is packaged as `examples/rope_sodium_formate.py`, which
saves `examples/output/rope_sodium_formate.png`.

## CROP Efficiency

CROP adds DD-CSA cross-correlated relaxation. `crop_eta(ka, kc, J_hz)` computes
the `Iz -> 2IzSz` efficiency for rates supplied in the same frequency units as
`J_hz`. The physical domain is `ka >= kc >= 0`.

```python
from optimalcontrol.crop import crop_eta, crop_eta_prime, crop_limit_Iz_to_Sz

J_hz = 100.0
ka = 60.0
kc = 45.0
ka_prime = 75.0
kc_prime = 50.0

eta = crop_eta(ka, kc, J_hz)
eta_prime = crop_eta_prime(ka_prime, kc_prime, J_hz)
two_step = crop_limit_Iz_to_Sz(eta, eta_prime)

print(f"eta = {eta:.3f}")
print(f"eta prime = {eta_prime:.3f}")
print(f"Iz -> Sz limit = {two_step:.3f}")
```

The limiting cases are useful sanity checks: `kc = 0` reduces to the ROPE
formula, and `kc -> ka` approaches lossless transfer.

## CROP Waveforms

`crop_pulse_params()` returns the scalar amplitude, carrier frequency, and
default truncation window used by `crop_waveform()`. CROP amplitude and
irradiation frequency are returned in Hz.

```python
from optimalcontrol.crop import crop_pulse_params, crop_waveform

J_hz = 100.0
ka = 0.6 * J_hz
kc = 0.75 * ka

params = crop_pulse_params(ka, kc, J_hz)
dt = params.truncation_window / 200.0
waveform = crop_waveform(
    ka=ka,
    kc=kc,
    J_hz=J_hz,
    dt=dt,
    truncation_window=params.truncation_window,
)

print(params)
print(waveform["times"].shape)
print(waveform["amplitude"][0], waveform["irrad_freq"][0])
```

Use a finite `truncation_window`; in the exact `kc == ka` lossless limit the
default window is infinite and must be replaced by an experimental pulse length.
