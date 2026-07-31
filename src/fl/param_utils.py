"""Flatten/unflatten a list of ndarrays (a model's parameters) to/from one 1D vector.

Used by the top-k sparsification strategy to select the top-k parameters *globally*
across the whole model (not per-layer) and to reconstruct dense updates server-side.
"""
import numpy as np


def get_shapes(ndarrays: list[np.ndarray]) -> list[tuple]:
    return [a.shape for a in ndarrays]


def flatten(ndarrays: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    sizes = [a.size for a in ndarrays]
    flat = np.concatenate([a.ravel() for a in ndarrays]).astype(np.float32)
    return flat, sizes


def unflatten(flat: np.ndarray, shapes: list[tuple], sizes: list[int]) -> list[np.ndarray]:
    out = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        out.append(flat[offset : offset + size].reshape(shape).astype(np.float32))
        offset += size
    return out


def dense_bytes(ndarrays: list[np.ndarray]) -> int:
    return int(sum(a.nbytes for a in ndarrays))
