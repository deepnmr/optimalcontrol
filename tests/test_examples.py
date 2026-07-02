"""Headless regression tests for all example scripts.

Each test imports an example module, calls run(), and asserts the output
matches the stored .npz snapshot within rtol=1e-4.
"""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import numpy.testing as npt
import pytest

_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
_EXPECTED_DIR = os.path.join(_EXAMPLES_DIR, "expected")

_EXAMPLE_NAMES = [
    "rope_efficiency_vs_n",
    "rope_gain_Ia_2IbSc",
    "rope_inphase_gain",
    "rope_finite_time_efficiency",
    "rope_finite_time_controls",
    "rope_sodium_formate",
    "rope_hard_pulse",
    "jmr2005_fig5_rope",
    "crop_efficiency_vs_ka",
    "crop_truncated_waveform",
    "crop_lossless_limit",
    "crop_regression_params",
    "grape_broadband_180",
    "phase_only_iy_inversion_10khz",
    "hmqc_oc_180_artifact",
    "methyl_water_binary_symmetric_180",
    "methyl_water_reburp_180",
    "methyl_water_reburp_minpower_180",
    "methyl_water_reburp_minlength_180",
    "reburp_pulse",
]


def _load_snapshot(name: str) -> np.ndarray:  # type: ignore[type-arg]
    path = os.path.join(_EXPECTED_DIR, f"{name}.npz")
    data = np.load(path)
    return data["output"]  # type: ignore[return-value]


def _import_example(name: str) -> object:
    import importlib
    import sys

    examples_parent = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
    if examples_parent not in sys.path:
        sys.path.insert(0, examples_parent)
    return importlib.import_module(f"examples.{name}")


@pytest.mark.parametrize("name", _EXAMPLE_NAMES)
def test_example_runs_without_exception(name: str) -> None:
    mod = _import_example(name)
    run = getattr(mod, "run")
    result = run()
    assert result is not None


@pytest.mark.parametrize("name", _EXAMPLE_NAMES)
def test_example_matches_snapshot(name: str) -> None:
    mod = _import_example(name)
    run = getattr(mod, "run")
    result = run()
    expected = _load_snapshot(name)
    npt.assert_allclose(
        result,
        expected,
        rtol=1e-4,
        atol=1e-12,
        err_msg=f"Example {name!r} output deviates from snapshot",
    )
