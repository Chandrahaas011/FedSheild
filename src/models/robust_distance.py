"""Lightweight class-conditional robust-distance novelty scoring."""
from __future__ import annotations

import numpy as np


class ClassConditionalRobustDistance:
    """Estimate candidate-class novelty with robust diagonal distances.

    Centres, scales, and empirical reference distributions are learned from training
    data only. Higher output values indicate greater novelty relative to a candidate
    class. The estimator is deliberately compact and deterministic.
    """

    def __init__(self, reference_rows_per_class: int = 50_000, seed: int = 0):
        if reference_rows_per_class < 100:
            raise ValueError("reference_rows_per_class must be at least 100")
        self.reference_rows_per_class = int(reference_rows_per_class)
        self.seed = int(seed)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    @staticmethod
    def _distance(X: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
        robust_z = np.abs((X - centre) / scale)
        np.clip(robust_z, 0.0, 1.0e6, out=robust_z)
        return np.log1p(robust_z).mean(axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._validate_X(X)
        y = np.asarray(y, dtype=np.int8)
        if len(X) != len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError("X and binary y must have compatible rows")
        centres, scales, references = [], [], []
        for label in (0, 1):
            class_X = X[y == label]
            if len(class_X) < 2:
                raise ValueError(f"class {label} requires at least two training rows")
            centre = np.median(class_X, axis=0).astype(np.float32)
            q25, q75 = np.percentile(class_X, [25, 75], axis=0)
            scale = ((q75 - q25) / 1.349).astype(np.float32)
            fallback = np.std(class_X, axis=0, dtype=np.float64).astype(np.float32)
            scale = np.where(scale > 1.0e-12, scale, fallback)
            scale = np.where(scale > 1.0e-12, scale, 1.0).astype(np.float32)

            rng = np.random.default_rng(self.seed + label * 1009)
            if len(class_X) > self.reference_rows_per_class:
                index = np.sort(
                    rng.choice(
                        len(class_X), self.reference_rows_per_class, replace=False
                    )
                )
                reference_X = class_X[index]
            else:
                reference_X = class_X
            reference = np.sort(self._distance(reference_X, centre, scale)).astype(
                np.float32
            )
            centres.append(centre)
            scales.append(scale)
            references.append(reference)
        self.centres_ = np.asarray(centres, dtype=np.float32)
        self.scales_ = np.asarray(scales, dtype=np.float32)
        self.reference_distances_ = references
        self.n_features_in_ = int(X.shape[1])
        return self

    def candidate_novelty(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        if not hasattr(self, "centres_"):
            raise ValueError("distance model is not fitted")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature count does not match fitted distance model")
        output = np.empty((len(X), 2), dtype=np.float32)
        for label in (0, 1):
            distances = self._distance(
                X, self.centres_[label], self.scales_[label]
            )
            reference = self.reference_distances_[label]
            ranks = np.searchsorted(reference, distances, side="right")
            output[:, label] = ranks / float(len(reference) + 1)
        return output
