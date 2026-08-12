"""Evaluation helpers for the frozen robust-distance abstention extension."""
from __future__ import annotations

import numpy as np

from experiments.uncertainty_common import evaluate_uncertainty
from src.evaluation.abstention import (
    fit_score_conformal_thresholds,
    fuse_candidate_scores,
    probability_nonconformity,
    score_conformal_metrics,
)


def evaluate_density_uncertainty(
    y_calibration: np.ndarray,
    calibration_probability: np.ndarray,
    calibration_novelty: np.ndarray,
    y_test: np.ndarray,
    test_probability: np.ndarray,
    test_novelty: np.ndarray,
    alpha: float,
    target_coverage: float,
    probability_weight: float,
) -> dict:
    """Compare probability, distance, and fixed-weight fused conformal scores."""
    base = evaluate_uncertainty(
        y_calibration,
        calibration_probability,
        y_test,
        test_probability,
        alpha,
        target_coverage,
    )
    probability_calibration_scores = probability_nonconformity(calibration_probability)
    probability_test_scores = probability_nonconformity(test_probability)
    fused_calibration_scores = fuse_candidate_scores(
        probability_calibration_scores, calibration_novelty, probability_weight
    )
    fused_test_scores = fuse_candidate_scores(
        probability_test_scores, test_novelty, probability_weight
    )

    distance_thresholds = fit_score_conformal_thresholds(
        y_calibration, calibration_novelty, alpha, class_conditional=True
    )
    fused_thresholds = fit_score_conformal_thresholds(
        y_calibration, fused_calibration_scores, alpha, class_conditional=True
    )
    methods = dict(base["methods"])
    methods["distance_mondrian_alpha_10"] = score_conformal_metrics(
        y_test, test_probability, test_novelty, distance_thresholds
    )
    methods["fused_distance_probability_mondrian_alpha_10"] = score_conformal_metrics(
        y_test, test_probability, fused_test_scores, fused_thresholds
    )

    fused_curve = []
    for curve_alpha in (0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01):
        thresholds = fit_score_conformal_thresholds(
            y_calibration,
            fused_calibration_scores,
            curve_alpha,
            class_conditional=True,
        )
        fused_curve.append(
            {
                "alpha": curve_alpha,
                "nominal_class_conditional_set_coverage": 1.0 - curve_alpha,
                **score_conformal_metrics(
                    y_test, test_probability, fused_test_scores, thresholds
                ),
            }
        )

    probability_method = methods["mondrian_conformal_alpha_10"]
    fused_method = methods["fused_distance_probability_mondrian_alpha_10"]
    base_fn = methods["no_abstention"]["confident_false_negative_rate"]
    fused_fn = fused_method["confident_false_negative_rate"]
    relative_reduction = (
        float((base_fn - fused_fn) / base_fn) if base_fn > 0 else None
    )
    improves_automation = (
        fused_method["automatic_malicious_detection_rate"]
        > probability_method["automatic_malicious_detection_rate"]
        or fused_method["malicious_rejection_rate"]
        < probability_method["malicious_rejection_rate"]
    )
    return {
        **base,
        "methods": methods,
        "curves": {**base["curves"], "fused_mondrian": fused_curve},
        "density_protocol": {
            "distance": "class-conditional mean log robust standardized deviation",
            "novelty_scaling": "training-only class-conditional empirical percentile",
            "probability_weight": float(probability_weight),
            "distance_weight": float(1.0 - probability_weight),
        },
        "density_success_criterion": {
            "confident_false_negative_relative_reduction": relative_reduction,
            "benign_rejection_rate": fused_method["benign_rejection_rate"],
            "improves_automation_over_probability_mondrian": bool(improves_automation),
            "target_relative_reduction": 0.25,
            "maximum_benign_rejection_rate": 0.15,
            "passed": bool(
                relative_reduction is not None
                and relative_reduction >= 0.25
                and fused_method["benign_rejection_rate"] <= 0.15
                and improves_automation
            ),
        },
    }
