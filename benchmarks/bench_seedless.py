"""Benchmark public Seedless objective calls across pulse lengths.

Run from the repository root:
    .venv/bin/python benchmarks/bench_seedless.py
    OPTIMALCONTROL_DISABLE_RUST=1 .venv/bin/python benchmarks/bench_seedless.py
"""

from statistics import median
from timeit import repeat

import numpy as np

from optimalcontrol._accelerator import _enabled
from optimalcontrol.ocseed import Band, SeedlessSpec


def main() -> None:
    backend = "rust" if _enabled() else "numpy"
    print("backend,restraint,steps,members,seconds_per_call")
    for steps in (32, 128, 512):
        phases = np.random.default_rng(7).uniform(-np.pi, np.pi, steps)
        for band in (
            Band(-1.0, 1.0, "s2s", n_offsets=7, init="z", targ="-z"),
            Band(-1.0, 1.0, "universal", n_offsets=7, rotation=("x", 180.0)),
            Band(-1.0, 1.0, "suppress", n_offsets=7, per_step=True),
        ):
            spec = SeedlessSpec(
                spectrometer_mhz=600.0,
                carrier_ppm=0.0,
                rf_max_hz=2500.0,
                duration_s=4e-3,
                n_steps=steps,
                bands=[band],
                b1_scales=(0.8, 1.0, 1.2),
            )
            spec.objective(phases)
            seconds = median(repeat(lambda: spec.objective(phases), number=1, repeat=5))
            print(f"{backend},{band.restraint},{steps},21,{seconds:.8e}", flush=True)


if __name__ == "__main__":
    main()
