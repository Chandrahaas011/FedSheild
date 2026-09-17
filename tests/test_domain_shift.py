from pathlib import Path

import joblib
import numpy as np
import pytest

from experiments.domain_shift_common import (
    label_blind_window_indices,
    label_blind_source_roles,
    mixed_windows,
    score_window_stream,
    score_windows,
)
from src.models.domain_shift import (
    BENIGN,
    DOMAIN_SHIFT_DETECTED,
    MALICIOUS,
    UNKNOWN,
    QuantileJSDShiftSentinel,
    MarginalShiftSentinel,
    ShiftGuardedClassifier,
)


class _FixedAbstainer:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.int8)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    def predict_with_rejection(self, X):
        return np.resize(self.predictions, len(X))


def _sentinel(seed=3):
    rng = np.random.default_rng(100)
    train = rng.normal(size=(4096, 4)).astype(np.float32)
    validation = rng.normal(size=(4096, 4)).astype(np.float32)
    sentinel = QuantileJSDShiftSentinel(
        window_size=128,
        quantile_bins=8,
        calibration_alpha=0.10,
        maximum_reference_rows=2048,
        minimum_calibration_windows=8,
        seed=seed,
    )
    return sentinel.fit(train).calibrate(validation), train, validation


def test_sentinel_is_deterministic_and_calibration_audit_are_disjoint():
    first, train, validation = _sentinel(seed=7)
    second = QuantileJSDShiftSentinel(
        window_size=128,
        quantile_bins=8,
        calibration_alpha=0.10,
        maximum_reference_rows=2048,
        minimum_calibration_windows=8,
        seed=7,
    ).fit(train).calibrate(validation)
    assert first.threshold_ == second.threshold_
    assert np.array_equal(first.calibration_scores_, second.calibration_scores_)
    assert np.array_equal(first.audit_scores_, second.audit_scores_)
    calibration_rows = set(first.calibration_window_indices_.ravel())
    audit_rows = set(first.audit_window_indices_.ravel())
    assert calibration_rows.isdisjoint(audit_rows)


def test_shifted_windows_score_higher_and_raise_batch_alarm():
    sentinel, train, _ = _sentinel()
    in_domain = train[:128]
    shifted = in_domain + 20.0
    assert sentinel.score_window(shifted) > sentinel.score_window(in_domain)
    assert sentinel.assess_window(shifted)["status"] == DOMAIN_SHIFT_DETECTED
    with pytest.raises(ValueError, match="exactly 128 rows"):
        sentinel.assess_window(shifted[:1])


def test_four_operational_outputs_and_serialization(tmp_path: Path):
    sentinel, train, _ = _sentinel()
    trusted = None
    for start in range(0, len(train), 128):
        candidate = train[start : start + 128]
        if len(candidate) == 128 and not sentinel.assess_window(candidate)["shift_detected"]:
            trusted = candidate
            break
    assert trusted is not None
    guarded = ShiftGuardedClassifier(_FixedAbstainer([0, 1, -1]), sentinel)
    trusted_outputs = guarded.predict_operational(trusted)
    assert {BENIGN, MALICIOUS, UNKNOWN}.issubset(set(trusted_outputs))
    shifted_outputs = guarded.predict_operational(trusted + 20.0)
    assert set(shifted_outputs) == {DOMAIN_SHIFT_DETECTED}
    path = tmp_path / "guard.joblib"
    joblib.dump(guarded, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.predict_operational(trusted), trusted_outputs)


def test_window_stream_is_label_blind_deterministic_and_reports_tail():
    sentinel, train, _ = _sentinel()
    X = np.vstack([train[:256], np.zeros((13, 4), dtype=np.float32)])
    first_indices, first_tail = label_blind_window_indices(len(X), 128, seed=11)
    second_indices, second_tail = label_blind_window_indices(len(X), 128, seed=11)
    assert np.array_equal(first_indices, second_indices)
    assert first_tail == second_tail == 13
    summary = score_window_stream(sentinel, X, seed=11)
    assert summary["windows"] == 2
    assert summary["rows_omitted_incomplete_window"] == 13


def test_comparable_shift_methods_use_disjoint_roles_and_detect_large_shift():
    rng = np.random.default_rng(33)
    source = rng.normal(size=(4096, 4)).astype(np.float32)
    target = (rng.normal(size=(4096, 4)) + 10).astype(np.float32)
    roles = label_blind_source_roles(len(source), 128, seed=5)
    role_sets = [set(index) for name, index in roles.items() if name != "omitted"]
    assert role_sets[0].isdisjoint(role_sets[1])
    assert role_sets[0].isdisjoint(role_sets[2])
    for method in ("jsd_mean", "psi_mean", "ks_mean"):
        detector = MarginalShiftSentinel(
            method=method,
            window_size=128,
            quantile_bins=8,
            calibration_alpha=0.10,
            maximum_reference_rows=2048,
            maximum_ks_reference_rows=512,
            minimum_calibration_windows=8,
            seed=5,
        ).fit(source).calibrate(source[roles["calibration"]], source[roles["audit"]])
        windows = mixed_windows(
            source[roles["mixture"]], target, 128, 0.5, 4, seed=8
        )
        assert score_windows(detector, windows)["alarm_rate"] == 1.0
