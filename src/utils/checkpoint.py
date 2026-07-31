"""Checkpoint/resume helpers.

Built in from day one per ROADMAP.md Phase 3: free-tier GPU sessions die mid-run, so
every training loop (centralized epochs, FL rounds) must be resumable from disk without
re-running completed work. Also makes local runs safe to Ctrl-C.
"""
import json
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: Path, step: int, model_state: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "model_state": model_state, "extra": extra or {}}
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def save_run_result(path: Path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


def load_run_result(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
