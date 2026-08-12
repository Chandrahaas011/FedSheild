from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class SelectedFeatureModel:
    """Serializable inference wrapper that enforces the trained feature order."""

    def __init__(self, estimator, feature_indices: list[int], feature_names: list[str]):
        self.estimator = estimator
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.feature_names = list(feature_names)

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.asarray(X)[:, self.feature_indices])

    def predict(self, X):
        return self.estimator.predict(np.asarray(X)[:, self.feature_indices])


class BenignEnvelopeFusionModel:
    """Compact hybrid of supervised probability and benign-envelope novelty."""

    def __init__(
        self,
        supervised_estimator,
        fusion_estimator,
        feature_indices: list[int],
        feature_names: list[str],
        benign_median: np.ndarray,
        benign_scale: np.ndarray,
        top_n: int = 3,
    ):
        self.supervised_estimator = supervised_estimator
        self.fusion_estimator = fusion_estimator
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.feature_names = list(feature_names)
        self.benign_median = np.asarray(benign_median, dtype=float)
        self.benign_scale = np.asarray(benign_scale, dtype=float)
        self.top_n = int(top_n)

    def novelty_score(self, selected_X: np.ndarray) -> np.ndarray:
        deviation = np.abs((selected_X - self.benign_median) / self.benign_scale)
        top_n = min(self.top_n, deviation.shape[1])
        largest = np.partition(deviation, deviation.shape[1] - top_n, axis=1)[:, -top_n:]
        return np.log1p(largest.mean(axis=1))

    def _fusion_features(self, X) -> np.ndarray:
        selected = np.asarray(X)[:, self.feature_indices]
        supervised_probability = self.supervised_estimator.predict_proba(selected)[:, 1]
        novelty = self.novelty_score(selected)
        return np.column_stack([supervised_probability, novelty])

    def predict_proba(self, X):
        return self.fusion_estimator.predict_proba(self._fusion_features(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build_lightweight_model(name: str, seed: int):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                average=True,
                random_state=seed,
            ),
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=18,
            min_samples_leaf=5,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    raise ValueError(f"Unknown lightweight model: {name}")
