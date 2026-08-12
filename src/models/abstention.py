"""Serializable wrappers for confidence and conformal abstention."""
from __future__ import annotations

import numpy as np

from src.evaluation.abstention import (
    confidence_predictions,
    conformal_prediction_sets,
    singleton_predictions,
    fuse_candidate_scores,
    probability_nonconformity,
    score_conformal_prediction_sets,
)


class AbstainingClassifier:
    def __init__(self, estimator, method: str, parameters):
        if method not in {"confidence", "conformal"}:
            raise ValueError("method must be confidence or conformal")
        self.estimator = estimator
        self.method = method
        self.parameters = parameters

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict_with_rejection(self, X) -> np.ndarray:
        positive = self.predict_proba(X)[:, 1]
        if self.method == "confidence":
            predictions, _ = confidence_predictions(positive, float(self.parameters))
            return predictions
        sets = conformal_prediction_sets(positive, np.asarray(self.parameters, dtype=float))
        predictions, _ = singleton_predictions(sets)
        return predictions


class DistanceAwareAbstainingClassifier:
    """Serializable probability-plus-distance conformal rejector."""

    def __init__(self, estimator, distance_model, thresholds, probability_weight=0.50):
        self.estimator = estimator
        self.distance_model = distance_model
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.probability_weight = float(probability_weight)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def candidate_scores(self, X) -> np.ndarray:
        positive = self.predict_proba(X)[:, 1]
        return fuse_candidate_scores(
            probability_nonconformity(positive),
            self.distance_model.candidate_novelty(X),
            self.probability_weight,
        )

    def predict_with_rejection(self, X) -> np.ndarray:
        sets = score_conformal_prediction_sets(
            self.candidate_scores(X), self.thresholds
        )
        predictions, _ = singleton_predictions(sets)
        return predictions
