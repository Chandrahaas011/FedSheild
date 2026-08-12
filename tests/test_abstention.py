import numpy as np

from src.evaluation.abstention import (
    confidence_metrics,
    confidence_threshold,
    conformal_metrics,
    finite_sample_quantile,
    fit_conformal_thresholds,
    fit_score_conformal_thresholds,
    fuse_candidate_scores,
    probability_nonconformity,
    score_conformal_metrics,
)


def test_finite_sample_quantile_uses_higher_corrected_rank():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert finite_sample_quantile(scores, alpha=0.25) == 0.4


def test_mondrian_thresholds_are_class_specific():
    y = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    thresholds = fit_conformal_thresholds(y, probability, alpha=0.34, class_conditional=True)
    assert thresholds.shape == (2,)
    assert np.isfinite(thresholds).all()


def test_conformal_rejects_ambiguous_probability():
    y_cal = np.array([0, 0, 1, 1])
    p_cal = np.array([0.1, 0.2, 0.8, 0.9])
    thresholds = fit_conformal_thresholds(y_cal, p_cal, alpha=0.25, class_conditional=True)
    metrics = conformal_metrics(np.array([0]), np.array([0.5]), thresholds)
    assert metrics["coverage"] == 0.0
    assert metrics["empty_set_rate"] == 1.0


def test_confidence_threshold_reaches_target_with_no_ties():
    probabilities = np.array([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99])
    threshold = confidence_threshold(probabilities, target_coverage=0.5)
    metrics = confidence_metrics(
        np.array([0, 0, 0, 0, 1, 1, 1, 1]), probabilities, threshold
    )
    assert metrics["coverage"] == 0.5


def test_abstention_reduces_confident_false_negatives():
    y = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.49, 0.9])
    metrics = confidence_metrics(y, probabilities, threshold=0.85)
    assert metrics["malicious_rejection_rate"] == 2 / 3
    assert metrics["confident_false_negative_rate"] == 0.0


def test_candidate_score_fusion_and_conformal_metrics():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.05, 0.20, 0.80, 0.95])
    novelty = np.array(
        [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]], dtype=float
    )
    fused = fuse_candidate_scores(
        probability_nonconformity(probability), novelty, probability_weight=0.5
    )
    thresholds = fit_score_conformal_thresholds(y, fused, alpha=0.25)
    metrics = score_conformal_metrics(y, probability, fused, thresholds)
    assert fused.shape == (4, 2)
    assert metrics["coverage"] == 1.0
    assert metrics["automatic_malicious_detection_rate"] == 1.0
    assert metrics["analyst_escalations_per_10000_flows"] == 0.0
