"""Microbenchmarks for matrix exponential helpers.

Run from the repository root with:

    python benchmarks/bench_expm.py
"""

import time

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from optimalcontrol.grape import dir_diff_expm

ComplexArray = npt.NDArray[np.complex128]


def _hermitian_matrix(dim: int, seed: int) -> ComplexArray:
    """Return a deterministic Hermitian matrix for timing."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((dim, dim))
    imag = rng.standard_normal((dim, dim))
    matrix = real + np.complex128(1j) * imag
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return np.asarray(hermitian, dtype=np.complex128)


def _time_call(fn: object, repeats: int) -> float:
    """Return average seconds per call."""
    callable_fn = fn
    start = time.perf_counter()
    for _ in range(repeats):
        callable_fn()
    elapsed = time.perf_counter() - start
    return elapsed / float(repeats)


def _benchmark_dim(dim: int) -> tuple[float, float]:
    """Return average timings for expm and dir_diff_expm at one dimension."""
    matrix = _hermitian_matrix(dim, seed=dim)
    direction = _hermitian_matrix(dim, seed=dim + 10_000)
    dt = 1e-3
    repeats = 300 if dim == 4 else 50

    expm_arg = np.complex128(-1j * dt) * matrix
    _ = expm(expm_arg)
    _ = dir_diff_expm(matrix, direction, dt)

    expm_seconds = _time_call(lambda: expm(expm_arg), repeats)
    frechet_seconds = _time_call(lambda: dir_diff_expm(matrix, direction, dt), repeats)
    return expm_seconds, frechet_seconds


def main() -> None:
    """Print timing results for the supported benchmark dimensions."""
    print("dim,expm_s,dir_diff_expm_s")
    for dim in (4, 16):
        expm_seconds, frechet_seconds = _benchmark_dim(dim)
        print(f"{dim},{expm_seconds:.8e},{frechet_seconds:.8e}")


if __name__ == "__main__":
    main()
