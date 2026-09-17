"""Small serializable models used by the bounded recovery study."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbabilityRecalibrator:
    """Platt-style recalibration of a frozen classifier's malicious probability."""

    def __init__(self, base_estimator, seed: int = 0):
        self.base_estimator = base_estimator
        self.seed = int(seed)

    @staticmethod
    def _logit_feature(probability: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probability, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
        return np.log(probability / (1.0 - probability)).reshape(-1, 1)

    def fit(self, X, y):
        probability = self.base_estimator.predict_proba(X)[:, 1]
        self.calibrator_ = LogisticRegression(
            class_weight="balanced", solver="lbfgs", random_state=self.seed
        ).fit(self._logit_feature(probability), np.asarray(y, dtype=np.int8))
        return self

    def predict_proba(self, X):
        probability = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.calibrator_.predict_proba(self._logit_feature(probability))[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
