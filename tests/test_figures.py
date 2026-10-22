"""Tests for the example figure helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from examples._figures import quantise, save_figure


def test_quantise_erases_sub_12_decimal_noise_but_keeps_1e6_scale() -> None:
    base = np.array([0.1234567890123456, -0.9999999999999])
    noisy = base + 1e-14
    npt.assert_array_equal(quantise(base), quantise(noisy))

    scaled = np.array([1.0e-6])
    npt.assert_array_equal(quantise(scaled), scaled)


def test_save_figure_writes_png_without_matplotlib_version(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    out = save_figure(fig, tmp_path / "tiny.png")
    data = out.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"Matplotlib version" not in data
