import numpy as np

from src.metrics import compute_metrics, summarize_over_seeds


def test_perfect_predictions_give_f1_one():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred, class_names=["A", "B", "C"])
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["overall_fpr"] == 0.0


def test_all_benign_predicted_as_attack_gives_fpr_one():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, class_names=["Benign", "Attack"], benign_idx=0)
    assert m["overall_fpr"] == 1.0


def test_summarize_over_seeds_mean_std():
    y_true = np.array([0, 1, 1, 0])
    m1 = compute_metrics(y_true, np.array([0, 1, 1, 0]), class_names=["Benign", "Attack"])
    m2 = compute_metrics(y_true, np.array([0, 0, 1, 0]), class_names=["Benign", "Attack"])
    summary = summarize_over_seeds([m1, m2])
    assert summary["accuracy_mean"] == (m1["accuracy"] + m2["accuracy"]) / 2
    assert "per_class_recall_mean" in summary
