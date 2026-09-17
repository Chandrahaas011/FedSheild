from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.safe_resumption_common import (
    allocate_recovery_roles,
    attach_review_label_noise,
    choose_resumption_action,
    gate_passes,
    nested_balanced_review,
    uncertainty_aware_gate_passes,
    wilson_upper_bound,
)
from src.models.recovery import ProbabilityRecalibrator
from src.models.lightweight import build_lightweight_model


def _target_frame(rows=10000):
    rng = np.random.default_rng(4)
    labels = np.asarray([0] * 7000 + [1] * 3000, dtype=np.int8)
    return pd.DataFrame({
        "feature": rng.normal(size=rows),
        "binary_label": labels,
        "feature_fingerprint": np.arange(rows, dtype=np.uint64),
    })


def test_recovery_roles_are_deterministic_disjoint_and_review_budgets_are_nested():
    first = allocate_recovery_roles(_target_frame(), 2000, 3000, 4000, seed=9)
    second = allocate_recovery_roles(_target_frame(), 2000, 3000, 4000, seed=9)
    for role in first:
        assert first[role]["recovery_original_index"].tolist() == second[role]["recovery_original_index"].tolist()
    role_sets = [set(frame["recovery_original_index"]) for frame in first.values()]
    assert role_sets[0].isdisjoint(role_sets[1])
    assert role_sets[0].isdisjoint(role_sets[2])
    assert role_sets[1].isdisjoint(role_sets[2])
    small = nested_balanced_review(first["review"], 100, seed=9)
    large = nested_balanced_review(first["review"], 500, seed=9)
    assert set(small["recovery_original_index"]) <= set(large["recovery_original_index"])
    assert small["binary_label"].value_counts().to_dict() == {0: 50, 1: 50}


def test_shadow_validation_role_is_disjoint_and_prevalence_preserving():
    roles = allocate_recovery_roles(
        _target_frame(), 1000, 2000, 3000, seed=12, shadow_validation_rows=2000
    )
    assert set(roles) == {"review", "validation", "shadow_validation", "audit"}
    index_sets = [set(frame["recovery_original_index"]) for frame in roles.values()]
    for left in range(len(index_sets)):
        for right in range(left + 1, len(index_sets)):
            assert index_sets[left].isdisjoint(index_sets[right])
    validation_rate = roles["validation"]["binary_label"].mean()
    shadow_rate = roles["shadow_validation"]["binary_label"].mean()
    assert abs(shadow_rate - validation_rate) < 0.001


def test_review_label_noise_is_deterministic_balanced_and_preserves_clean_labels():
    review = allocate_recovery_roles(_target_frame(), 2000, 2000, 3000, seed=4)["review"]
    first = attach_review_label_noise(review, 0.10, seed=31)
    second = attach_review_label_noise(review, 0.10, seed=31)
    assert first["binary_label"].equals(review["binary_label"])
    assert first["recovery_training_label"].equals(second["recovery_training_label"])
    assert int(first["review_label_corrupted"].sum()) == 200
    corrupted = first[first["review_label_corrupted"]]
    assert corrupted["binary_label"].value_counts().to_dict() == {0: 100, 1: 100}
    assert (corrupted["binary_label"] != corrupted["recovery_training_label"]).all()


def test_robust_tree_has_frozen_noise_tolerant_capacity():
    model = build_lightweight_model("robust_decision_tree", seed=7)
    assert model.max_depth == 8
    assert model.min_samples_leaf == 20
    assert model.ccp_alpha == 0.001


def test_probability_recalibrator_is_serializable(tmp_path: Path):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] > 0).astype(np.int8)
    base = LogisticRegression().fit(X, y)
    model = ProbabilityRecalibrator(base, seed=2).fit(X[:200], y[:200])
    before = model.predict_proba(X[200:220])
    path = tmp_path / "recalibrator.joblib"
    joblib.dump(model, path)
    after = joblib.load(path).predict_proba(X[200:220])
    assert np.allclose(before, after)
    assert np.allclose(before.sum(axis=1), 1.0)


def test_gate_and_selection_do_not_use_final_audit():
    metrics = {
        "macro_f1": 0.9,
        "false_positive_rate": 0.02,
        "ece_15": 0.03,
        "per_class": {"Malicious": {"recall": 0.9}},
    }
    efficiency = {
        "serialized_model_bytes": 1000,
        "single_record_latency_ms_p95": 1.0,
    }
    criteria = {
        "minimum_macro_f1": 0.8,
        "minimum_malicious_recall": 0.8,
        "maximum_false_positive_rate": 0.1,
        "maximum_ece": 0.1,
        "maximum_model_bytes": 2000,
        "maximum_single_record_latency_p95_ms": 2.0,
    }
    assert gate_passes(metrics, efficiency, criteria)
    actions = [
        {"action": "compact_retraining", "label_budget": 500, "gate_passed": True,
         "efficiency": efficiency, "validation_metrics": metrics,
         "final_audit_metrics": {"macro_f1": 0.0}},
        {"action": "recalibration", "label_budget": 100, "gate_passed": True,
         "efficiency": efficiency, "validation_metrics": metrics,
         "final_audit_metrics": {"macro_f1": 1.0}},
    ]
    selected = choose_resumption_action(actions)
    assert selected["action"] == "recalibration"
    assert selected["label_budget"] == 100


def test_uncertainty_aware_gate_rejects_boundary_fpr_without_audit_data():
    metrics = {
        "macro_f1": 0.92,
        "false_positive_rate": 236 / 2645,
        "ece_15": 0.05,
        "confusion_matrix": [[2409, 236], [385, 6970]],
        "per_class": {"Malicious": {"recall": 6970 / 7355}},
    }
    efficiency = {
        "serialized_model_bytes": 4071,
        "single_record_latency_ms_p95": 0.06,
    }
    criteria = {
        "minimum_macro_f1": 0.8,
        "minimum_malicious_recall": 0.8,
        "maximum_false_positive_rate": 0.1,
        "maximum_ece": 0.1,
        "maximum_model_bytes": 1048576,
        "maximum_single_record_latency_p95_ms": 5.0,
    }
    assert gate_passes(metrics, efficiency, criteria)
    passed, upper = uncertainty_aware_gate_passes(metrics, efficiency, criteria)
    assert not passed
    assert upper > 0.1
    assert wilson_upper_bound(0, 100) > 0.0


def test_uncertainty_aware_gate_accepts_clear_fpr_margin():
    metrics = {
        "macro_f1": 0.91,
        "false_positive_rate": 50 / 2500,
        "ece_15": 0.04,
        "confusion_matrix": [[2450, 50], [300, 7200]],
        "per_class": {"Malicious": {"recall": 0.96}},
    }
    efficiency = {
        "serialized_model_bytes": 5000,
        "single_record_latency_ms_p95": 0.1,
    }
    criteria = {
        "minimum_macro_f1": 0.8,
        "minimum_malicious_recall": 0.8,
        "maximum_false_positive_rate": 0.1,
        "maximum_ece": 0.1,
        "maximum_model_bytes": 1048576,
        "maximum_single_record_latency_p95_ms": 5.0,
    }
    passed, upper = uncertainty_aware_gate_passes(metrics, efficiency, criteria)
    assert passed
    assert metrics["false_positive_rate"] < upper < 0.1


def test_class_conditional_review_noise_flips_only_requested_clean_class():
    frame = _target_frame()
    noisy = attach_review_label_noise(
        frame, 0.0, seed=13, rates_by_clean_label={0: 0.10, 1: 0.0}
    )
    benign = noisy["binary_label"] == 0
    malicious = ~benign
    assert int(noisy.loc[benign, "review_label_corrupted"].sum()) == 700
    assert not noisy.loc[malicious, "review_label_corrupted"].any()
    assert (noisy.loc[benign & noisy["review_label_corrupted"], "recovery_training_label"] == 1).all()
    assert (noisy.loc[malicious, "recovery_training_label"] == 1).all()
