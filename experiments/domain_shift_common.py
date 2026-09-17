"""Common scoring and resource measurements for the domain-shift sentinel."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.evaluation.efficiency import process_memory_mb
from src.models.domain_shift import (
    BENIGN,
    DOMAIN_SHIFT_DETECTED,
    MALICIOUS,
    UNKNOWN,
    ShiftGuardedClassifier,
)


def label_blind_window_indices(
    row_count: int, window_size: int, seed: int
) -> tuple[np.ndarray, int]:
    if row_count < window_size:
        raise ValueError("at least one complete window is required")
    rng = np.random.default_rng(seed)
    complete_rows = (row_count // window_size) * window_size
    selected = rng.permutation(row_count)[:complete_rows]
    return selected.reshape(-1, window_size), int(row_count - complete_rows)


def label_blind_source_roles(
    row_count: int, window_size: int, seed: int
) -> dict[str, np.ndarray]:
    """Create disjoint source calibration, audit, and mixture pools without labels."""
    total_windows = row_count // window_size
    if total_windows < 12:
        raise ValueError("at least 12 complete source-validation windows are required")
    rng = np.random.default_rng(seed + 8201)
    selected = rng.permutation(row_count)[: total_windows * window_size]
    windows = selected.reshape(total_windows, window_size)
    calibration_count = total_windows // 2
    remaining = total_windows - calibration_count
    audit_count = remaining // 2
    mixture_count = remaining - audit_count
    if audit_count < 1 or mixture_count < 1:
        raise ValueError("source roles require non-empty audit and mixture pools")
    return {
        "calibration": windows[:calibration_count].ravel(),
        "audit": windows[calibration_count : calibration_count + audit_count].ravel(),
        "mixture": windows[calibration_count + audit_count :].ravel(),
        "omitted": np.setdiff1d(np.arange(row_count), selected, assume_unique=False),
    }


def mixed_windows(
    source_X: np.ndarray,
    target_X: np.ndarray,
    window_size: int,
    target_fraction: float,
    windows: int,
    seed: int,
) -> np.ndarray:
    """Build label-blind fixed-size source/target mixtures without replacement per role."""
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in [0, 1]")
    if windows < 1:
        raise ValueError("at least one mixture window is required")
    target_rows_per_window = int(round(window_size * target_fraction))
    source_rows_per_window = window_size - target_rows_per_window
    source_needed = windows * source_rows_per_window
    target_needed = windows * target_rows_per_window
    if len(source_X) < source_needed or len(target_X) < target_needed:
        raise ValueError("insufficient rows for the requested label-blind mixture")
    rng = np.random.default_rng(seed)
    source_indices = rng.permutation(len(source_X))[:source_needed]
    target_indices = rng.permutation(len(target_X))[:target_needed]
    output = np.empty((windows, window_size, source_X.shape[1]), dtype=np.float32)
    source_cursor = target_cursor = 0
    for window_index in range(windows):
        rows = []
        if source_rows_per_window:
            rows.append(source_X[source_indices[source_cursor : source_cursor + source_rows_per_window]])
            source_cursor += source_rows_per_window
        if target_rows_per_window:
            rows.append(target_X[target_indices[target_cursor : target_cursor + target_rows_per_window]])
            target_cursor += target_rows_per_window
        window = np.vstack(rows)
        output[window_index] = window[rng.permutation(len(window))]
    return output


def score_windows(sentinel, windows: np.ndarray) -> dict:
    if windows.ndim != 3 or windows.shape[1] != sentinel.window_size:
        raise ValueError("windows must have shape (n, sentinel.window_size, features)")
    scores = np.asarray([sentinel.score_window(window) for window in windows])
    alarms = scores > sentinel.threshold_
    return {
        "windows": int(len(windows)),
        "alarm_count": int(alarms.sum()),
        "alarm_rate": float(alarms.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "scores": [float(value) for value in scores],
    }


def score_window_stream(sentinel, X: np.ndarray, seed: int) -> dict:
    windows, omitted = label_blind_window_indices(
        len(X), sentinel.window_size, seed
    )
    scores = np.asarray([sentinel.score_window(X[index]) for index in windows])
    alarms = scores > sentinel.threshold_
    alarm_positions = np.flatnonzero(alarms)
    return {
        "windows": int(len(windows)),
        "rows_scored": int(windows.size),
        "rows_omitted_incomplete_window": omitted,
        "alarm_count": int(alarms.sum()),
        "alarm_rate": float(alarms.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "score_median": float(np.median(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "first_alarm_window": int(alarm_positions[0]) if len(alarm_positions) else None,
        "first_alarm_after_rows": (
            int((alarm_positions[0] + 1) * sentinel.window_size)
            if len(alarm_positions)
            else None
        ),
        "scores": [float(value) for value in scores],
    }


def source_audit_summary(sentinel) -> dict:
    scores = np.asarray(sentinel.audit_scores_, dtype=np.float64)
    alarms = scores > sentinel.threshold_
    return {
        "windows": int(len(scores)),
        "rows_scored": int(len(scores) * sentinel.window_size),
        "alarm_count": int(alarms.sum()),
        "alarm_rate": float(alarms.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "score_median": float(np.median(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "scores": [float(value) for value in scores],
    }


def evaluate_guarded_stream(
    guarded_model: ShiftGuardedClassifier,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict:
    y = np.asarray(y, dtype=np.int8)
    windows, omitted = label_blind_window_indices(
        len(X), guarded_model.shift_sentinel.window_size, seed
    )
    counts = {BENIGN: 0, MALICIOUS: 0, UNKNOWN: 0, DOMAIN_SHIFT_DETECTED: 0}
    base_confident_false_negatives = 0
    withheld_base_confident_false_negatives = 0
    remaining_confident_false_negatives = 0
    malicious_rows = 0
    for index in windows:
        window_X = X[index]
        window_y = y[index]
        base = guarded_model.abstaining_classifier.predict_with_rejection(window_X)
        outputs = guarded_model.predict_operational(window_X)
        for status in counts:
            counts[status] += int((outputs == status).sum())
        malicious = window_y == 1
        base_fn = malicious & (base == 0)
        shifted = outputs[0] == DOMAIN_SHIFT_DETECTED
        base_confident_false_negatives += int(base_fn.sum())
        if shifted:
            withheld_base_confident_false_negatives += int(base_fn.sum())
        else:
            remaining_confident_false_negatives += int(base_fn.sum())
        malicious_rows += int(malicious.sum())
    scored = int(windows.size)
    return {
        "windows": int(len(windows)),
        "rows_scored": scored,
        "rows_omitted_incomplete_window": omitted,
        "output_counts": counts,
        "output_rates": {key: float(value / scored) for key, value in counts.items()},
        "base_confident_false_negatives": base_confident_false_negatives,
        "withheld_base_confident_false_negatives": withheld_base_confident_false_negatives,
        "remaining_confident_false_negatives": remaining_confident_false_negatives,
        "malicious_rows": malicious_rows,
        "base_confident_false_negative_rate": (
            float(base_confident_false_negatives / malicious_rows)
            if malicious_rows else 0.0
        ),
        "remaining_confident_false_negative_rate": (
            float(remaining_confident_false_negatives / malicious_rows)
            if malicious_rows else 0.0
        ),
        "withheld_fraction_of_base_confident_false_negatives": (
            float(withheld_base_confident_false_negatives / base_confident_false_negatives)
            if base_confident_false_negatives else 0.0
        ),
        "interpretation": (
            "DOMAIN_SHIFT_DETECTED is a deployment interlock, not a malicious "
            "prediction; withheld errors are not counted as detected attacks."
        ),
    }


def benchmark_shift_guard(
    guarded_model: ShiftGuardedClassifier,
    X_in_domain: np.ndarray,
    artifact_path: Path,
    repeats: int = 50,
) -> dict:
    window_size = guarded_model.shift_sentinel.window_size
    if len(X_in_domain) < window_size:
        raise ValueError("benchmark requires one complete in-domain window")
    window = np.asarray(X_in_domain[:window_size])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(guarded_model, artifact_path, compress=3)
    if guarded_model.shift_sentinel.assess_window(window)["shift_detected"]:
        raise ValueError("benchmark window must pass the source-calibrated shift guard")
    guarded_model.predict_operational(window)
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        guarded_model.predict_operational(window)
        times.append((time.perf_counter_ns() - start) / 1e6)
    memory = process_memory_mb()
    total_seconds = sum(times) / 1000.0
    return {
        "measurement": "batch_shift_assessment_plus_conditional_conformal_decision",
        "serialized_guarded_model_bytes": int(artifact_path.stat().st_size),
        "window_size": int(window_size),
        "window_latency_ms_median": float(np.median(times)),
        "window_latency_ms_p95": float(np.percentile(times, 95)),
        "throughput_records_per_second": float(window_size * repeats / total_seconds),
        **memory,
    }
