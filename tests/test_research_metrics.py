import numpy as np

from src.evaluation.research_metrics import binary_metrics, expected_calibration_error, select_thresholds


def test_binary_metric_values_for_perfect_probabilities():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.1, 0.9, 0.99])
    metrics = binary_metrics(y, probabilities)
    assert metrics["macro_f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert expected_calibration_error(y, probabilities) >= 0.0


def test_selected_fpr_threshold_respects_constraint_when_feasible():
    y = np.array([0] * 100 + [1] * 20)
    probabilities = np.array([0.01] * 99 + [0.8] + [0.9] * 20)
    thresholds = select_thresholds(y, probabilities, fpr_constraint=0.0)
    metrics = binary_metrics(y, probabilities, thresholds["validation_fpr_constrained"])
    assert metrics["false_positive_rate"] == 0.0

