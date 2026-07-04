# Creating Bruker Pulses with optimalcontrol

This repository supports two broad ways to create pulses:

1. Sample an analytical solution directly: `optimalcontrol.rope.rope_waveform()`, `optimalcontrol.crop.crop_waveform()`
2. Solve a numerical optimal-control problem: `optimalcontrol.grape.ControlProblem` + GRAPE

If you want a `.shape` file that you can take straight to Bruker, the best main example in this repository is [`examples/sciadv2023_fig1_ur180.py`](examples/sciadv2023_fig1_ur180.py). It builds a low-power UR-180 pulse with phase-only GRAPE and writes both a Bruker shape and a diagnostic figure.

## Setup

The examples in this document were validated by running them from the repository root.

- Python `3.13.13`
- SciPy `1.17.1`
- Matplotlib `3.10.9`

The example scripts import the package directly, so this setup is sufficient:

```bash
python3 -m pip install scipy
python3 -m pip install matplotlib
python3 -m pip install -e .
```

If you want the full development environment, this is enough:

```bash
pip install -e ".[dev]"
```

## Main Example: Generate a `sciadv2023` Bruker Shape

The fastest way to run the example is:

```bash
python3 -m examples.sciadv2023_fig1_ur180
```

Actual output from a verified run:

```text
Saved Bruker shape /home2/dlee/project/optimalcontrol/examples/output/sciadv2023_fig1_ur180.shape
Saved figure /home2/dlee/project/optimalcontrol/examples/output/sciadv2023_fig1_ur180.png
```

Generated files:

- `examples/output/sciadv2023_fig1_ur180.shape`
- `examples/output/sciadv2023_fig1_ur180.png`

Pulse parameters in this example:

- `N_STEPS = 72`
- Total duration `540 us`
- Step duration `7.5 us`
- Nominal RF `7.5 kHz`
- Offset bandwidth `+/-6.3 kHz`
- B1 compensation range `+/-15 %`

In other words, this example exports a UR-180 pulse with constant amplitude and time-varying phase as a Bruker shape.

## What Is Inside the Bruker Shape

The generated `.shape` file contains a header like this:

```text
##TITLE= sciadv2023_fig1_ur180
##$SHAPE_TOTROT= 1.800000e+02
##$SHAPE_INTEGFAC= 1.000000e+00
##$SHAPE_MODE= 1
##$OPTIMALCONTROL_TOTAL_DURATION_S= 5.400000000000e-04
##$OPTIMALCONTROL_STEP_DURATION_S= 7.500000000000e-06
##$OPTIMALCONTROL_RF_HZ= 7.500000000000e+03
##$OPTIMALCONTROL_BANDWIDTH_HZ= 6.300000000000e+03
##$OPTIMALCONTROL_B1_DEVIATION_PERCENT= 1.500000000000e+01
##$OPTIMALCONTROL_NOTE= Set pulse length to TOTAL_DURATION_S and calibrate 100% to RF_HZ.
##NPOINTS= 72
##XYPOINTS= (XY..XY)
1.000000000e+02, 6.598765774e+00
1.000000000e+02, 1.723134983e+01
...
```

How to read it:

- The first column is amplitude in `%`
- The second column is phase in `degree`
- This pulse uses constant amplitude, so the amplitude stays at `100` for every point

The key settings you still need on the Bruker side are:

- Pulse length: `540 us`
- RF calibration such that amplitude `100%` corresponds to `7.5 kHz`

So the `.shape` file alone is not the whole story. The spectrometer settings have to match these assumptions.

## Calling It Directly from Python

You can also call the example directly from Python instead of launching it as a script.

```python
from examples.sciadv2023_fig1_ur180 import run

result = run(optimize=False)
phase_deg = result[:72]
summary = result[-6:]
```

`result` is an array of length `78`.

- The first `72` values are the phase samples for each time slice. They are in degrees and wrapped to `0..360`.
- The final `6` values are summary metrics. The order is mean transfer efficiency, transfer-efficiency standard deviation, mean `Mxy`, `Mxy` standard deviation, mean phase error, and phase-error standard deviation.

The last 6 values from the verified run were:

```text
[0.997738, 0.00322, 0.999055, 0.001462, 0.090669, 2.942966]
```

## Re-optimizing the Pulse

By default, the example uses the cached phase solution stored in the repository. If you want to rerun the phase-only GRAPE optimization and regenerate the shape, use:

```bash
python3 -m examples.sciadv2023_fig1_ur180 --optimize
```

This path takes longer, but it is the standard way to regenerate the pulse from the current code.

## Minimal GRAPE Example for Understanding the Structure

The `sciadv2023` example is the most useful one for actual shape generation, but a much smaller `ControlProblem` is easier if you want to understand the internal data flow. The example below optimizes the smallest XY pulse that drives a single-spin transfer `Iz -> Ix`.

```python
from pathlib import Path

import numpy as np

from optimalcontrol.grape import ControlProblem, grape_xy
from optimalcontrol.io import export_bruker, export_csv, export_json
from optimalcontrol.operators import Ix, Iy, Iz
from optimalcontrol.optimizers import run_grape
from optimalcontrol.states import normalise_hs

output_dir = Path("examples/output/minimal_grape_pulse")
output_dir.mkdir(parents=True, exist_ok=True)

cp = ControlProblem(
    drifts=[np.zeros((2, 2), dtype=np.complex128)],
    operators=[np.complex128(-1j) * Ix(), np.complex128(-1j) * Iy()],
    rho_init=[normalise_hs(Iz())],
    rho_targ=[normalise_hs(Ix())],
    pulse_dt=0.1,
    pwr_levels=[1.0, 1.0],
    freeze=None,
    fidelity_mode="real",
    basis="dense",
)

wfm0 = np.zeros((4, 2), dtype=np.float64)
initial_fidelity = grape_xy(cp, wfm0)
waveform, result = run_grape(cp, wfm0, method="lbfgs", m=4, tol_x=0.0, tol_g=0.0, max_iter=20)

export_json(waveform, output_dir / "pulse.json")
export_csv(waveform, output_dir / "pulse.csv")
export_bruker(waveform, output_dir / "pulse.shape")

print(f"initial fidelity: {initial_fidelity:.6f}")
print(f"final fidelity:   {result.fidelity_final:.6f}")
print(f"iterations:       {result.n_iter}")
print(f"reason:           {result.reason}")
print("internal shape:", result.wfm_final.shape)
print("export shape:  ", waveform.data.shape)
print("final waveform:")
print(result.wfm_final)
```

Actual output from a verified run:

```text
initial fidelity: 0.000000
final fidelity:   1.000000
iterations:       6
reason:           step_tol
internal shape: (4, 2)
export shape:   (2, 4)
final waveform:
[[0.         3.92699082]
 [0.         3.92699082]
 [0.         3.92699082]
 [0.         3.92699082]]
```

What matters in this example:

- Internal waveform layout is `(n_steps, n_channels)`
- Exported `Waveform.data` layout is `(n_channels, n_steps)`
- `run_grape()` returns both the optimization result and an export-ready waveform

That said, the `export_bruker()` used here is only a minimal compatibility stub. For a more realistic spectrometer-facing example, `sciadv2023_fig1_ur180.py` remains the better reference.

## How to Extend to Larger Problems

Starting from either the `sciadv2023` example or the minimal example, these are the main fields you usually change:

1. `drifts`: add chemical shift, J-coupling, or relaxation
2. `operators`: expand to the real control channels, for example `Ix`, `Iy`, `Sx`, `Sy`
3. `rho_init`, `rho_targ`: define the transfer you want, for example `Iz -> 2IzSz`
4. `n_steps`, `pulse_dt`: adjust pulse duration and time resolution
5. `offsets`, `offset_operators`, `pwr_levels`: add broadband robustness, offset robustness, and B1 robustness

For larger GRAPE examples, see [`examples/grape_broadband_180.py`](examples/grape_broadband_180.py), [`examples/jmr2005_fig5_rope.py`](examples/jmr2005_fig5_rope.py), and [`tests/test_integration.py`](tests/test_integration.py).

## If You Want Analytical Pulses Instead

If you want to sample a waveform directly instead of optimizing one numerically, use the analytical APIs:

```python
from optimalcontrol.crop import crop_pulse_params, crop_waveform
from optimalcontrol.rope import rope_waveform

rope = rope_waveform(T=0.263 / 100.0, n=1.0, J_hz=100.0, dt=(0.263 / 100.0) / 400.0)
ka = 0.6 * 100.0
kc = 0.75 * ka
params = crop_pulse_params(ka, kc, J_hz=100.0)
crop = crop_waveform(ka, kc, J_hz=100.0, dt=1e-4, truncation_window=params.truncation_window)
```

That path means "sample a pulse whose formula is already known", while GRAPE means "search for a pulse that achieves the target transfer".
