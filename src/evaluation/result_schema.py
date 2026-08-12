from __future__ import annotations


REQUIRED_TOP_LEVEL = {
    "experiment_id",
    "created_at",
    "status",
    "code_version",
    "dataset",
    "protocol",
    "seed",
    "model",
    "feature_set",
    "metrics",
    "efficiency",
    "durations",
    "hardware",
}


def validate_result(result: dict) -> None:
    missing = REQUIRED_TOP_LEVEL - set(result)
    if missing:
        raise ValueError(f"Result is missing required fields: {sorted(missing)}")
    if result["status"] not in {"complete", "failed"}:
        raise ValueError("Result status must be 'complete' or 'failed'")
    if result["status"] == "complete":
        for key in ("macro_f1", "mcc", "false_positive_rate", "brier_score"):
            if key not in result["metrics"]:
                raise ValueError(f"Completed result is missing metric: {key}")
        if result["feature_set"]["count"] <= 0:
            raise ValueError("Feature count must be positive")

