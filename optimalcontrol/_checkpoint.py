"""Optimizer checkpoint persistence and the shared optimizer result type."""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optimalcontrol._lbfgs import LBFGSState as LBFGSState
from optimalcontrol._lbfgs import _copy_lbfgs_state as _copy_lbfgs_state
from optimalcontrol._line_search import _as_real_array as _as_real_array
from optimalcontrol._problem import ControlProblem as ControlProblem
from optimalcontrol._problem import validate_waveform as validate_waveform
from optimalcontrol._types import Array, RealArray


@dataclass
class OptimResult:
    """Common output contract for waveform optimizers."""

    wfm_final: RealArray
    fidelity_final: float
    n_iter: int
    n_feval: int
    converged: bool
    reason: str
    history: list[float]
    trajectory: list[Array] | None = None


@dataclass(frozen=True)
class _CheckpointData:
    """Validated checkpoint contents used by the optimizers."""

    wfm: RealArray
    history: list[float]
    n_feval: int = 0
    lbfgs_state: LBFGSState | None = None
    signature: str | None = None


def _initial_waveform(cp: ControlProblem, wfm0: RealArray) -> RealArray:
    """Return a validated mutable copy of the initial GRAPE waveform."""
    waveform = np.asarray(wfm0, dtype=np.float64)
    validate_waveform(waveform, len(cp.operators), waveform.shape[0] if waveform.ndim else 0)
    return waveform.copy()


def _validate_checkpoint_path(path: str) -> Path:
    """Return a usable checkpoint file path."""
    if not path.strip():
        raise ValueError("checkpoint_path must be a non-empty string")
    checkpoint_path = Path(path)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise ValueError("checkpoint_path must point to a file, not a directory")
    return checkpoint_path


def _checkpoint_signature(cp: ControlProblem, optimizer_name: str) -> str:
    """Return a signature binding a checkpoint to its problem and optimizer."""
    from optimalcontrol.io import _hash_control_problem

    payload = f"{optimizer_name}\0{_hash_control_problem(cp)}".encode()
    return hashlib.sha256(payload).hexdigest()


def _serialise_lbfgs_state(state: LBFGSState) -> dict[str, object]:
    """Return a JSON-compatible snapshot of an L-BFGS state."""
    memory = _copy_lbfgs_state(state)
    return {
        "m": int(memory["m"]),
        "s_history": [step.tolist() for step in memory["s_history"]],
        "y_history": [curvature.tolist() for curvature in memory["y_history"]],
        "rho_history": [float(rho) for rho in memory["rho_history"]],
    }


def _deserialise_lbfgs_state(raw: object) -> LBFGSState:
    """Return a validated L-BFGS state restored from checkpoint JSON."""
    if not isinstance(raw, dict):
        raise ValueError("checkpoint lbfgs_state must be a JSON object")

    try:
        m_value = int(raw["m"])
        s_raw = raw["s_history"]
        y_raw = raw["y_history"]
        rho_raw = raw["rho_history"]
    except KeyError as exc:
        raise ValueError(f"checkpoint lbfgs_state is missing {exc.args[0]!r}") from exc

    if not isinstance(s_raw, list) or not isinstance(y_raw, list) or not isinstance(rho_raw, list):
        raise ValueError("checkpoint lbfgs_state histories must be JSON lists")

    s_history = [
        _as_real_array("checkpoint s_history", np.asarray(value, dtype=np.float64))
        for value in s_raw
    ]
    y_history = [
        _as_real_array("checkpoint y_history", np.asarray(value, dtype=np.float64))
        for value in y_raw
    ]
    rho_history = [float(value) for value in rho_raw]
    if not all(math.isfinite(value) for value in rho_history):
        raise ValueError("checkpoint rho_history entries must be finite")

    state: LBFGSState = {
        "m": m_value,
        "s_history": s_history,
        "y_history": y_history,
        "rho_history": rho_history,
    }
    return _copy_lbfgs_state(state)


def _checkpoint_history(result_so_far: list[float] | OptimResult) -> list[float]:
    """Return the fidelity-history portion of a saveable result."""
    raw_history = result_so_far.history if isinstance(result_so_far, OptimResult) else result_so_far
    history = [float(value) for value in raw_history]
    if not all(math.isfinite(value) for value in history):
        raise ValueError("checkpoint history entries must be finite")
    return history


def _write_checkpoint(checkpoint_path: Path, checkpoint: _CheckpointData) -> None:
    """Write one optimizer checkpoint to disk."""
    waveform = _as_real_array("wfm", checkpoint.wfm)
    if waveform.ndim != 2:
        raise ValueError(f"wfm must be two-dimensional, got shape {waveform.shape}")

    history = _checkpoint_history(checkpoint.history)
    payload: dict[str, object] = {
        "wfm": waveform.tolist(),
        "history": history,
        "n_feval": int(checkpoint.n_feval),
    }
    if checkpoint.n_feval < 0:
        raise ValueError("checkpoint n_feval must be non-negative")
    if checkpoint.lbfgs_state is not None:
        payload["lbfgs_state"] = _serialise_lbfgs_state(checkpoint.lbfgs_state)
    if checkpoint.signature is not None:
        if not checkpoint.signature:
            raise ValueError("checkpoint signature must be a non-empty string")
        payload["signature"] = checkpoint.signature

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: fsync a temp file then os.replace, so an interrupted write
    # (Ctrl-C, OOM, power loss) cannot destroy the previous good checkpoint.
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, checkpoint_path)


def _read_checkpoint(path: str | Path) -> _CheckpointData:
    """Return validated checkpoint contents from disk."""
    checkpoint_path = path if isinstance(path, Path) else _validate_checkpoint_path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a JSON object")

    if "wfm" not in payload or "history" not in payload:
        raise ValueError("checkpoint payload must contain 'wfm' and 'history'")

    waveform = _as_real_array("checkpoint wfm", np.asarray(payload["wfm"], dtype=np.float64))
    if waveform.ndim != 2:
        raise ValueError(f"checkpoint wfm must be two-dimensional, got shape {waveform.shape}")

    history_value = payload["history"]
    if not isinstance(history_value, list):
        raise ValueError("checkpoint history must be a JSON list")
    history = _checkpoint_history(history_value)

    raw_n_feval = payload.get("n_feval", 0)
    n_feval = int(raw_n_feval)
    if n_feval < 0:
        raise ValueError("checkpoint n_feval must be non-negative")

    lbfgs_state = (
        _deserialise_lbfgs_state(payload["lbfgs_state"]) if "lbfgs_state" in payload else None
    )
    raw_signature = payload.get("signature")
    if raw_signature is not None and (not isinstance(raw_signature, str) or not raw_signature):
        raise ValueError("checkpoint signature must be a non-empty string")
    return _CheckpointData(
        wfm=waveform,
        history=history,
        n_feval=n_feval,
        lbfgs_state=lbfgs_state,
        signature=raw_signature,
    )


def _effective_checkpoint_path(
    cp: ControlProblem,
    checkpoint_path: str | None,
) -> Path | None:
    """Return the checkpoint file requested by the optimizer call, if any."""
    candidate = checkpoint_path if checkpoint_path is not None else cp.checkpoint_path
    if candidate is None:
        return None
    return _validate_checkpoint_path(candidate)


def _restore_checkpoint(
    cp: ControlProblem,
    wfm0: RealArray,
    checkpoint_path: Path | None,
    signature: str | None,
) -> _CheckpointData:
    """Return the starting optimizer state, loading a checkpoint if present."""
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = _read_checkpoint(checkpoint_path)
        waveform = _initial_waveform(cp, checkpoint.wfm)
        requested_shape = np.asarray(wfm0).shape
        if waveform.shape != requested_shape:
            raise ValueError(
                f"checkpoint waveform shape {waveform.shape} must match "
                f"wfm0 shape {requested_shape}"
            )
        if checkpoint.signature is None:
            return _CheckpointData(wfm=waveform, history=[], signature=signature)
        if checkpoint.signature != signature:
            raise ValueError("checkpoint optimizer or control problem does not match current run")
        return _CheckpointData(
            wfm=waveform,
            history=checkpoint.history.copy(),
            n_feval=checkpoint.n_feval,
            lbfgs_state=checkpoint.lbfgs_state,
            signature=checkpoint.signature,
        )

    waveform = _initial_waveform(cp, wfm0)
    return _CheckpointData(wfm=waveform, history=[], n_feval=0, lbfgs_state=None)


def _save_optimizer_checkpoint(
    checkpoint_path: Path | None,
    waveform: RealArray,
    history: list[float],
    *,
    n_feval: int,
    lbfgs_state: LBFGSState | None = None,
    signature: str | None = None,
) -> None:
    """Persist optimizer state when checkpointing is enabled."""
    if checkpoint_path is None:
        return
    _write_checkpoint(
        checkpoint_path,
        _CheckpointData(
            wfm=waveform,
            history=history,
            n_feval=n_feval,
            lbfgs_state=lbfgs_state,
            signature=signature,
        ),
    )


def save_checkpoint(path: str, wfm: RealArray, result_so_far: list[float] | OptimResult) -> None:
    """Save a waveform and fidelity history to a JSON checkpoint file."""
    checkpoint_path = _validate_checkpoint_path(path)
    n_feval = result_so_far.n_feval if isinstance(result_so_far, OptimResult) else 0
    _write_checkpoint(
        checkpoint_path,
        _CheckpointData(
            wfm=wfm,
            history=_checkpoint_history(result_so_far),
            n_feval=n_feval,
            lbfgs_state=None,
        ),
    )


def load_checkpoint(path: str) -> tuple[RealArray, list[float]]:
    """Load a waveform and fidelity history from a JSON checkpoint file."""
    checkpoint = _read_checkpoint(path)
    return checkpoint.wfm.copy(), checkpoint.history.copy()


def _validate_lbfgs_checkpoint_state(state: LBFGSState, waveform_shape: tuple[int, ...]) -> None:
    """Raise ValueError if restored L-BFGS arrays do not match the waveform shape."""
    memory = _copy_lbfgs_state(state)
    for name, history_list in (
        ("s_history", memory["s_history"]),
        ("y_history", memory["y_history"]),
    ):
        for index, array in enumerate(history_list):
            if array.shape != waveform_shape:
                raise ValueError(
                    f"checkpoint {name}[{index}] shape {array.shape} "
                    f"must match waveform shape {waveform_shape}"
                )
