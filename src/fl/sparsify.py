"""Top-k gradient/update sparsification with error-feedback (Lin et al., Deep Gradient
Compression, 2018) — a client sends only its largest-magnitude update coordinates each
round and accumulates the rest into a local residual to be sent later, so no update mass
is lost, only delayed.
"""
import numpy as np


def topk_sparsify(delta: np.ndarray, ratio: float, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (indices, values, new_residual). ratio in (0, 1]."""
    combined = delta + residual
    k = max(1, int(round(len(combined) * ratio)))
    if k >= len(combined):
        idx = np.arange(len(combined))
    else:
        idx = np.argpartition(-np.abs(combined), k - 1)[:k]
    idx = np.sort(idx)
    values = combined[idx]
    new_residual = combined.copy()
    new_residual[idx] = 0.0
    return idx.astype(np.int64), values.astype(np.float32), new_residual.astype(np.float32)


def residual_adaptive_topk_sparsify(
    delta: np.ndarray,
    residual: np.ndarray,
    min_ratio: float,
    max_ratio: float,
    energy_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Error-feedback sparsification with a per-update, energy-based budget.

    The client sends the smallest number of largest-magnitude coordinates whose squared
    L2 energy reaches ``energy_target`` of ``delta + residual``.  ``min_ratio`` and
    ``max_ratio`` bound that choice, preventing empty messages and uncontrolled bursts.
    The returned ratio is the *actual* selected fraction and is recorded as telemetry.
    """
    if not (0 < min_ratio <= max_ratio <= 1):
        raise ValueError("Require 0 < min_ratio <= max_ratio <= 1")
    if not (0 < energy_target <= 1):
        raise ValueError("energy_target must be in (0, 1]")

    combined = delta + residual
    n = len(combined)
    min_k = max(1, int(np.ceil(n * min_ratio)))
    max_k = max(min_k, int(np.ceil(n * max_ratio)))
    squared = np.square(combined, dtype=np.float64)
    total_energy = float(squared.sum())

    if total_energy == 0.0:
        k = min_k
        idx = np.arange(k)
    else:
        order = np.argsort(-np.abs(combined), kind="stable")
        cumulative_energy = np.cumsum(squared[order])
        required_k = int(np.searchsorted(cumulative_energy, energy_target * total_energy, side="left")) + 1
        k = min(max(required_k, min_k), max_k)
        idx = order[:k]

    idx = np.sort(idx)
    values = combined[idx]
    new_residual = combined.copy()
    new_residual[idx] = 0.0
    return idx.astype(np.int64), values.astype(np.float32), new_residual.astype(np.float32), k / n


def densify(idx: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    out[idx] = values
    return out


def sparse_bytes(idx: np.ndarray, values: np.ndarray) -> int:
    return int(idx.nbytes + values.nbytes)
