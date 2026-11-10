"""Freeze the public import surface: every pre-refactor import path must still resolve."""

import importlib
import json
from pathlib import Path

import optimalcontrol

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "import_surface.json"

PUBLIC_ALL = [
    "__version__",
    "set_random_seed",
    "Ix",
    "Iy",
    "Iz",
    "comm",
    "liouvillian_comm",
    "SpinSystem",
    "two_spin_system",
    "state_from_label",
    "fidelity_real",
    "rope_g",
    "rope_waveform",
    "crop_eta",
    "crop_waveform",
    "CROPPulse",
    "run_grape",
    "propagate_bloch_ensemble",
    "Band",
    "SeedlessSpec",
]


def _load_surface() -> dict[str, list[str]]:
    with FIXTURE.open(encoding="utf-8") as f:
        surface: dict[str, list[str]] = json.load(f)
    return surface


def test_every_pre_refactor_import_path_still_resolves() -> None:
    misses: list[str] = []
    for module, names in _load_surface().items():
        imported = importlib.import_module(module)
        for name in names:
            if not hasattr(imported, name):
                misses.append(f"{module}.{name}")
    assert not misses, f"public import paths no longer resolve: {misses}"


def test_public_all_is_frozen() -> None:
    assert optimalcontrol.__all__ == PUBLIC_ALL


def test_fixture_is_substantial() -> None:
    surface = _load_surface()
    total_names = sum(len(names) for names in surface.values())
    assert len(surface) >= 8
    assert total_names >= 60
