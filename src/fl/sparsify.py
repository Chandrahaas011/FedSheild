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


def densify(idx: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    out[idx] = values
    return out


def sparse_bytes(idx: np.ndarray, values: np.ndarray) -> int:
    return int(idx.nbytes + values.nbytes)
