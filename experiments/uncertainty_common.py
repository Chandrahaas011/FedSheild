from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation.abstention import (
    confidence_metrics,
    confidence_threshold,
    conformal_metrics,
    fit_conformal_thresholds,
    selective_metrics,
)
from src.evaluation.research_metrics import binary_metrics


def validation_roles(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Create disjoint 40/30/30 selection/calibration/rejection roles."""
    labels = np.asarray(labels, dtype=np.int8)
    indices = np.arange(len(labels))
    selection, remainder = train_test_split(
        indices, train_size=0.40, random_state=seed + 3109, stratify=labels
    )
    probability_calibration, rejection_calibration = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed + 3119,
        stratify=labels[remainder],
    )
    output = {
        "model_selection": np.sort(np.asarray(selection)),
        "probability_calibration": np.sort(np.asarray(probability_calibration)),
        "rejection_calibration": np.sort(np.asarray(rejection_calibration)),
    }
    combined = np.concatenate(list(output.values()))
    if len(np.unique(combined)) != len(labels):
        raise AssertionError("validation roles must be disjoint and exhaustive")
    return output


def _no_abstention_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    predictions = (np.asarray(probability) >= 0.5).astype(np.int8)
    return selective_metrics(
        y_true,
        probability,
        predictions=predictions,
        accepted=np.ones(len(y_true), dtype=bool),
    )


def evaluate_uncertainty(
    y_rejection_calibration: np.ndarray,
    rejection_calibration_probability: np.ndarray,
    y_test: np.ndarray,
    test_probability: np.ndarray,
    alpha: float,
    target_coverage: float,
) -> dict:
    confidence_cutoff = confidence_threshold(
        rejection_calibration_probability, target_coverage
    )
    global_thresholds = fit_conformal_thresholds(
        y_rejection_calibration,
        rejection_calibration_probability,
        alpha=alpha,
        class_conditional=False,
    )
    mondrian_thresholds = fit_conformal_thresholds(
        y_rejection_calibration,
        rejection_calibration_probability,
        alpha=alpha,
        class_conditional=True,
    )
    methods = {
        "no_abstention": _no_abstention_metrics(y_test, test_probability),
        "confidence_target_90": confidence_metrics(
            y_test, test_probability, confidence_cutoff
        ),
        "global_conformal_alpha_10": conformal_metrics(
            y_test, test_probability, global_thresholds
        ),
        "mondrian_conformal_alpha_10": conformal_metrics(
            y_test, test_probability, mondrian_thresholds
        ),
    }

    confidence_curve = []
    for coverage in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00):
        threshold = confidence_threshold(rejection_calibration_probability, coverage)
        confidence_curve.append(
            {
                "target_validation_coverage": coverage,
                **confidence_metrics(y_test, test_probability, threshold),
            }
        )
    mondrian_curve = []
    for curve_alpha in (0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01):
        thresholds = fit_conformal_thresholds(
            y_rejection_calibration,
            rejection_calibration_probability,
            alpha=curve_alpha,
            class_conditional=True,
        )
        mondrian_curve.append(
            {
                "alpha": curve_alpha,
                "nominal_class_conditional_set_coverage": 1.0 - curve_alpha,
                **conformal_metrics(y_test, test_probability, thresholds),
            }
        )

    base_fn = methods["no_abstention"]["confident_false_negative_rate"]
    conformal_fn = methods["mondrian_conformal_alpha_10"][
        "confident_false_negative_rate"
    ]
    reduction = (
        float((base_fn - conformal_fn) / base_fn) if base_fn > 0 else None
    )
    benign_rejection = methods["mondrian_conformal_alpha_10"][
        "benign_rejection_rate"
    ]
    return {
        "alpha": float(alpha),
        "target_validation_coverage": float(target_coverage),
        "calibration_rows": int(len(y_rejection_calibration)),
        "calibration_class_counts": {
            str(label): int((np.asarray(y_rejection_calibration) == label).sum())
            for label in (0, 1)
        },
        "methods": methods,
        "curves": {
            "confidence": confidence_curve,
            "mondrian_conformal": mondrian_curve,
        },
        "success_criterion": {
            "confident_false_negative_relative_reduction": reduction,
            "benign_rejection_rate": benign_rejection,
            "target_relative_reduction": 0.25,
            "maximum_benign_rejection_rate": 0.15,
            "passed": bool(
                reduction is not None
                and reduction >= 0.25
                and benign_rejection <= 0.15
            ),
        },
        "conformal_scope_note": (
            "Finite-sample class-conditional coverage assumes exchangeability; no "
            "coverage guarantee is claimed under campaign or held-out-family shift."
        ),
    }


def primary_binary_metrics(y_test: np.ndarray, test_probability: np.ndarray) -> dict:
    return binary_metrics(y_test, test_probability, threshold=0.5)
