"""Leakage-safe abstention and split-conformal classification utilities."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


UNKNOWN_LABEL = -1


def probability_matrix(positive_probability: np.ndarray) -> np.ndarray:
    positive_probability = np.asarray(positive_probability, dtype=float)
    if positive_probability.ndim != 1:
        raise ValueError("positive_probability must be one-dimensional")
    if not np.isfinite(positive_probability).all():
        raise ValueError("probabilities must be finite")
    if ((positive_probability < 0) | (positive_probability > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return np.column_stack([1.0 - positive_probability, positive_probability])


def probability_nonconformity(positive_probability: np.ndarray) -> np.ndarray:
    """Return candidate-class nonconformity scores with shape ``(n, 2)``."""
    return 1.0 - probability_matrix(positive_probability)


def fuse_candidate_scores(
    probability_scores: np.ndarray,
    novelty_scores: np.ndarray,
    probability_weight: float = 0.50,
) -> np.ndarray:
    probability_scores = np.asarray(probability_scores, dtype=float)
    novelty_scores = np.asarray(novelty_scores, dtype=float)
    if probability_scores.shape != novelty_scores.shape or probability_scores.ndim != 2:
        raise ValueError("candidate score matrices must have equal two-dimensional shape")
    if probability_scores.shape[1] != 2:
        raise ValueError("binary candidate score matrices must have two columns")
    if not 0.0 <= probability_weight <= 1.0:
        raise ValueError("probability_weight must lie in [0, 1]")
    if not np.isfinite(probability_scores).all() or not np.isfinite(novelty_scores).all():
        raise ValueError("candidate scores must be finite")
    return probability_weight * probability_scores + (1.0 - probability_weight) * novelty_scores


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal 'higher' quantile with the finite-sample (n+1) correction."""
    scores = np.sort(np.asarray(scores, dtype=float))
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if len(scores) == 0:
        raise ValueError("at least one calibration score is required")
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha)) - 1
    return float(scores[min(max(rank, 0), len(scores) - 1)])


def fit_conformal_thresholds(
    y_calibration: np.ndarray,
    positive_probability: np.ndarray,
    alpha: float = 0.10,
    class_conditional: bool = True,
) -> np.ndarray:
    y_calibration = np.asarray(y_calibration, dtype=int)
    probabilities = probability_matrix(positive_probability)
    if len(y_calibration) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    if not np.isin(y_calibration, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")
    true_scores = 1.0 - probabilities[np.arange(len(y_calibration)), y_calibration]
    if class_conditional:
        thresholds = []
        for label in (0, 1):
            class_scores = true_scores[y_calibration == label]
            thresholds.append(finite_sample_quantile(class_scores, alpha))
        return np.asarray(thresholds, dtype=float)
    threshold = finite_sample_quantile(true_scores, alpha)
    return np.asarray([threshold, threshold], dtype=float)


def fit_score_conformal_thresholds(
    y_calibration: np.ndarray,
    candidate_scores: np.ndarray,
    alpha: float = 0.10,
    class_conditional: bool = True,
) -> np.ndarray:
    y_calibration = np.asarray(y_calibration, dtype=int)
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    if candidate_scores.shape != (len(y_calibration), 2):
        raise ValueError("candidate_scores must have shape (len(y), 2)")
    if not np.isfinite(candidate_scores).all():
        raise ValueError("candidate scores must be finite")
    if not np.isin(y_calibration, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")
    true_scores = candidate_scores[np.arange(len(y_calibration)), y_calibration]
    if class_conditional:
        return np.asarray(
            [
                finite_sample_quantile(true_scores[y_calibration == label], alpha)
                for label in (0, 1)
            ],
            dtype=float,
        )
    threshold = finite_sample_quantile(true_scores, alpha)
    return np.asarray([threshold, threshold], dtype=float)


def score_conformal_prediction_sets(
    candidate_scores: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if candidate_scores.ndim != 2 or candidate_scores.shape[1] != 2:
        raise ValueError("candidate_scores must have shape (n, 2)")
    if thresholds.shape != (2,):
        raise ValueError("binary conformal thresholds must have shape (2,)")
    return candidate_scores <= thresholds.reshape(1, 2)


def conformal_prediction_sets(
    positive_probability: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    probabilities = probability_matrix(positive_probability)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (2,):
        raise ValueError("binary conformal thresholds must have shape (2,)")
    return (1.0 - probabilities) <= thresholds.reshape(1, 2)


def singleton_predictions(prediction_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    if prediction_sets.ndim != 2 or prediction_sets.shape[1] != 2:
        raise ValueError("binary prediction sets must have shape (n, 2)")
    accepted = prediction_sets.sum(axis=1) == 1
    predictions = np.full(len(prediction_sets), UNKNOWN_LABEL, dtype=np.int8)
    predictions[accepted] = prediction_sets[accepted].argmax(axis=1).astype(np.int8)
    return predictions, accepted


def confidence_threshold(
    positive_probability: np.ndarray, target_coverage: float = 0.90
) -> float:
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must lie in (0, 1]")
    probabilities = probability_matrix(positive_probability)
    confidence = probabilities.max(axis=1)
    ordered = np.sort(confidence)[::-1]
    index = max(0, math.ceil(target_coverage * len(ordered)) - 1)
    return float(ordered[index])


def confidence_predictions(
    positive_probability: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = probability_matrix(positive_probability)
    accepted = probabilities.max(axis=1) >= float(threshold)
    predictions = probabilities.argmax(axis=1).astype(np.int8)
    predictions[~accepted] = UNKNOWN_LABEL
    return predictions, accepted


def selective_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    predictions: np.ndarray,
    accepted: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = probability_matrix(positive_probability)
    base_predictions = probabilities.argmax(axis=1).astype(np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    accepted = np.asarray(accepted, dtype=bool)
    if not (len(y_true) == len(predictions) == len(accepted)):
        raise ValueError("all selective metric inputs must have equal length")

    accepted_y = y_true[accepted]
    accepted_predictions = predictions[accepted]
    accepted_errors = accepted_predictions != accepted_y
    base_errors = base_predictions != y_true
    rejected = ~accepted
    benign = y_true == 0
    malicious = y_true == 1
    accepted_count = int(accepted.sum())

    if accepted_count:
        selective_accuracy = float(accuracy_score(accepted_y, accepted_predictions))
        selective_macro_f1 = float(
            f1_score(accepted_y, accepted_predictions, labels=[0, 1], average="macro", zero_division=0)
        )
        selective_mcc = (
            float(matthews_corrcoef(accepted_y, accepted_predictions))
            if len(np.unique(accepted_y)) > 1 and len(np.unique(accepted_predictions)) > 1
            else 0.0
        )
        selective_error = float(accepted_errors.mean())
    else:
        selective_accuracy = selective_macro_f1 = selective_mcc = selective_error = None

    def rate(mask: np.ndarray, denominator_mask: np.ndarray) -> float:
        denominator = int(denominator_mask.sum())
        return float(mask.sum() / denominator) if denominator else 0.0

    return {
        "coverage": float(accepted.mean()),
        "rejection_rate": float(rejected.mean()),
        "accepted_count": accepted_count,
        "rejected_count": int(rejected.sum()),
        "selective_accuracy": selective_accuracy,
        "selective_error": selective_error,
        "selective_macro_f1": selective_macro_f1,
        "selective_mcc": selective_mcc,
        "utility_correct_accepted_rate": float(((predictions == y_true) & accepted).mean()),
        "benign_rejection_rate": rate(rejected & benign, benign),
        "malicious_rejection_rate": rate(rejected & malicious, malicious),
        "confident_false_negative_rate": rate(
            accepted & malicious & (predictions == 0), malicious
        ),
        "accepted_false_positive_rate": rate(
            accepted & benign & (predictions == 1), benign
        ),
        "security_alert_recall": 1.0
        - rate(accepted & malicious & (predictions == 0), malicious),
        "security_alert_false_positive_rate": rate(
            (rejected | (accepted & (predictions == 1))) & benign, benign
        ),
        "base_error_rejection_rate": rate(rejected & base_errors, base_errors),
        "rejection_precision_for_base_errors": (
            float((rejected & base_errors).sum() / rejected.sum()) if rejected.any() else 0.0
        ),
        "automatic_malicious_detection_rate": rate(
            accepted & malicious & (predictions == 1), malicious
        ),
        "analyst_escalations_per_10000_flows": float(rejected.mean() * 10_000),
        "benign_escalations_per_10000_benign": float(
            rate(rejected & benign, benign) * 10_000
        ),
        "malicious_escalations_per_10000_malicious": float(
            rate(rejected & malicious, malicious) * 10_000
        ),
        "confident_misses_per_10000_malicious": float(
            rate(accepted & malicious & (predictions == 0), malicious) * 10_000
        ),
    }


def conformal_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    prediction_sets = conformal_prediction_sets(positive_probability, thresholds)
    predictions, accepted = singleton_predictions(prediction_sets)
    output = selective_metrics(
        y_true, positive_probability, predictions=predictions, accepted=accepted
    )
    set_sizes = prediction_sets.sum(axis=1)
    output.update(
        {
            "set_coverage": float(
                prediction_sets[np.arange(len(y_true)), y_true].mean()
            ),
            "mean_set_size": float(set_sizes.mean()),
            "empty_set_rate": float((set_sizes == 0).mean()),
            "ambiguous_set_rate": float((set_sizes == 2).mean()),
            "singleton_set_rate": float((set_sizes == 1).mean()),
            "thresholds": [float(value) for value in thresholds],
        }
    )
    return output


def score_conformal_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    candidate_scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    prediction_sets = score_conformal_prediction_sets(candidate_scores, thresholds)
    predictions, accepted = singleton_predictions(prediction_sets)
    output = selective_metrics(
        y_true, positive_probability, predictions=predictions, accepted=accepted
    )
    set_sizes = prediction_sets.sum(axis=1)
    output.update(
        {
            "set_coverage": float(
                prediction_sets[np.arange(len(y_true)), y_true].mean()
            ),
            "mean_set_size": float(set_sizes.mean()),
            "empty_set_rate": float((set_sizes == 0).mean()),
            "ambiguous_set_rate": float((set_sizes == 2).mean()),
            "singleton_set_rate": float((set_sizes == 1).mean()),
            "thresholds": [float(value) for value in thresholds],
        }
    )
    return output


def confidence_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    threshold: float,
) -> dict:
    predictions, accepted = confidence_predictions(positive_probability, threshold)
    output = selective_metrics(
        y_true, positive_probability, predictions=predictions, accepted=accepted
    )
    output["confidence_threshold"] = float(threshold)
    return output
