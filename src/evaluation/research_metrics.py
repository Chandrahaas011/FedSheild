from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def binary_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "ece_15": expected_calibration_error(y_true, probabilities, n_bins=15),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            "Benign": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "Malicious": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
    }


def select_thresholds(
    y_true: np.ndarray, probabilities: np.ndarray, fpr_constraint: float = 0.01
) -> dict[str, float]:
    candidates = np.unique(
        np.concatenate(
            [np.linspace(0.01, 0.99, 99), np.quantile(probabilities, np.linspace(0.01, 0.99, 99))]
        )
    )
    scored = [(float(value), binary_metrics(y_true, probabilities, float(value))) for value in candidates]
    best_f1 = max(scored, key=lambda item: (item[1]["macro_f1"], -item[0]))
    feasible = [item for item in scored if item[1]["false_positive_rate"] <= fpr_constraint]
    constrained = max(
        feasible,
        key=lambda item: (item[1]["macro_f1"], item[1]["per_class"]["Malicious"]["recall"]),
    ) if feasible else (1.0, binary_metrics(y_true, probabilities, 1.0))
    return {
        "default": 0.5,
        "validation_macro_f1": best_f1[0],
        "validation_fpr_constrained": constrained[0],
    }

