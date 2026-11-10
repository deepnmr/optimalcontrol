"""Benchmark the Bloch ensemble forward model across ensemble sizes.

Run from the repository root:
    .venv/bin/python benchmarks/bench_bloch.py
    OPTIMALCONTROL_DISABLE_RUST=1 .venv/bin/python benchmarks/bench_bloch.py
"""

from functools import partial
from statistics import median
from timeit import repeat

import numpy as np

from optimalcontrol._accelerator import _enabled
from optimalcontrol.bloch import propagate_bloch_ensemble


def main() -> None:
    backend = "rust" if _enabled() else "numpy"
    print("backend,steps,offsets,b1,seconds_per_call")
    for steps, n_offsets, n_b1 in ((64, 101, 1), (512, 801, 1), (512, 801, 3), (5136, 2001, 1)):
        waveform = np.random.default_rng(7).uniform(-1.0, 1.0, (steps, 2))
        offsets = np.linspace(-10.0, 10.0, n_offsets) * 1000.0
        scales = np.linspace(0.8, 1.2, n_b1)
        initial = np.array([0.0, 0.0, 1.0])

        call = partial(
            propagate_bloch_ensemble, initial, waveform, offsets, scales, 6000.0, 1.0 / steps
        )
        call()
        seconds = median(repeat(call, number=1, repeat=3))
        print(f"{backend},{steps},{n_offsets},{n_b1},{seconds:.8e}", flush=True)


if __name__ == "__main__":
    main()
