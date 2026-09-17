import pytest

from src.evaluation.result_schema import validate_result


def test_result_schema_rejects_incomplete_records():
    with pytest.raises(ValueError):
        validate_result({"status": "complete"})


def test_result_schema_rejects_domain_shift_record_without_shift_metrics():
    record = {
        "experiment_id": "domain_shift_example",
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "complete",
        "code_version": {},
        "dataset": {},
        "protocol": {},
        "seed": 0,
        "model": {},
        "feature_set": {"count": 1},
        "metrics": {
            "macro_f1": 0.5,
            "mcc": 0.0,
            "false_positive_rate": 0.5,
            "brier_score": 0.25,
        },
        "efficiency": {},
        "durations": {},
        "hardware": {},
    }
    with pytest.raises(ValueError, match="domain_shift"):
        validate_result(record)
