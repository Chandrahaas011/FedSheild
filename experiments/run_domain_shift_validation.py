"""Validate source-only drift sentinels on label-blind NetFlow-v2 samples."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.domain_shift_common import (
    label_blind_source_roles,
    mixed_windows,
    score_window_stream,
    score_windows,
    source_audit_summary,
)
from experiments.research_common import git_state, hardware_metadata, save_result, utc_now
from experiments.run_cross_netflow import DATASET_KEYS, _cap_label_blind, signed_log1p
from src.models.domain_shift import MarginalShiftSentinel


def load_label_blind_partitions(config: dict, source: str, target: str, seed: int):
    """Use no labels for canonical capping, role allocation, or target sampling."""
    if source == target or source not in DATASET_KEYS or target not in DATASET_KEYS:
        raise ValueError("source and target must be distinct supported datasets")
    processed = Path(config["processed_dir"])
    source_frame = pd.read_parquet(processed / f"{source}.parquet")
    target_frame = pd.read_parquet(processed / f"{target}.parquet")
    source_hashes = set(source_frame["feature_fingerprint"].astype("uint64"))
    overlap = target_frame["feature_fingerprint"].astype("uint64").isin(source_hashes)
    target_frame = target_frame[~overlap].reset_index(drop=True)

    rng = np.random.default_rng(seed + 9101)
    source_order = rng.permutation(len(source_frame))
    train_count = int(np.floor(0.70 * len(source_frame)))
    train = source_frame.iloc[np.sort(source_order[:train_count])].reset_index(drop=True)
    validation = source_frame.iloc[np.sort(source_order[train_count:])].reset_index(drop=True)
    return {
        "train": _cap_label_blind(train, config.get("maximum_train_rows"), seed + 9111),
        "validation": _cap_label_blind(validation, config.get("maximum_validation_rows"), seed + 9121),
        "target": _cap_label_blind(target_frame, config.get("maximum_target_test_rows"), seed + 9131),
    }, int(overlap.sum())


def _sentinel(settings: dict, method: str, seed: int) -> MarginalShiftSentinel:
    return MarginalShiftSentinel(
        method=method,
        window_size=int(settings["window_size"]),
        quantile_bins=int(settings["quantile_bins"]),
        calibration_alpha=float(settings["calibration_alpha"]),
        aggregation="mean",
        maximum_reference_rows=int(settings["maximum_reference_rows"]),
        maximum_ks_reference_rows=int(settings["maximum_ks_reference_rows"]),
        minimum_calibration_windows=int(settings["minimum_calibration_windows"]),
        seed=seed,
    )


def run(config: dict, source: str, target: str, seed: int, force: bool = False) -> dict:
    settings = config["domain_shift_validation"]
    experiment_prefix = str(settings.get("experiment_prefix", "domain_validation"))
    window_suffix = (
        f"_window{int(settings['window_size'])}"
        if experiment_prefix != "domain_validation" else ""
    )
    experiment_id = f"{experiment_prefix}{window_suffix}_{source}_to_{target}_seed{seed}"
    output_dir = Path(config["output_dir"])
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(Path(config["metadata_path"]).read_text(encoding="utf-8"))
    feature_names = metadata["common_feature_columns"]
    load_start = time.perf_counter()
    parts, overlap_removed = load_label_blind_partitions(config, source, target, seed)
    arrays = {
        role: signed_log1p(frame[feature_names].to_numpy(dtype=np.float64))
        for role, frame in parts.items()
    }
    load_seconds = time.perf_counter() - load_start
    roles = label_blind_source_roles(
        len(arrays["validation"]), int(settings["window_size"]), seed
    )
    X_train = arrays["train"]
    X_calibration = arrays["validation"][roles["calibration"]]
    X_audit = arrays["validation"][roles["audit"]]
    X_mixture_source = arrays["validation"][roles["mixture"]]
    X_target = arrays["target"]

    methods = {}
    run_start = time.perf_counter()
    for method in settings["comparison_methods"]:
        detector = _sentinel(settings, method, seed).fit(X_train).calibrate(
            X_calibration, X_audit
        )
        mixtures = {}
        for fraction in settings["mixture_fractions"]:
            windows = mixed_windows(
                X_mixture_source,
                X_target,
                detector.window_size,
                float(fraction),
                int(settings["mixture_windows_per_fraction"]),
                seed + 9201 + int(round(float(fraction) * 100)),
            )
            mixtures[str(fraction)] = score_windows(detector, windows)
        methods[method] = {
            "method": method,
            "threshold": float(detector.threshold_),
            "reference_rows": detector.reference_rows_,
            "calibration_windows": int(len(detector.calibration_scores_)),
            "source_audit": source_audit_summary(detector),
            "target_full": score_window_stream(detector, X_target, seed + 9301),
            "mixtures": mixtures,
        }
    duration = time.perf_counter() - run_start
    primary = methods[settings["primary_method"]]
    primary_passed = bool(
        primary["source_audit"]["alarm_rate"]
        <= float(settings["maximum_source_audit_false_alarm_rate"])
        and primary["target_full"]["alarm_rate"] >= 0.80
        and primary["mixtures"]["0.5"]["alarm_rate"] >= 0.80
    )

    source_meta = metadata["datasets"][source]
    target_meta = metadata["datasets"][target]
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": "UQ NetFlow-v2 global-label-blind shift validation",
            "source": source,
            "target": target,
            "raw_archive_sha256": {
                "source": source_meta["archive_sha256"],
                "target": target_meta["archive_sha256"],
            },
            "rows": {name: int(len(frame)) for name, frame in parts.items()},
            "class_counts": {
                name: {
                    str(label): int(count)
                    for label, count in frame["binary_label"].value_counts().items()
                }
                for name, frame in parts.items()
            },
        },
        "protocol": {
            "name": str(settings.get(
                "protocol_name", "label_blind_shift_detector_comparison"
            )),
            "base_protocol": "cross_dataset_unsupervised_shift_validation",
            "source": source,
            "target": target,
            "sampling_seed": seed,
            "canonical_sampling": "global feature-hash cap independent of labels",
            "source_role_allocation": "label-blind 70/30 source split; validation split into calibration/audit/mixture roles without labels",
            "cross_dataset_overlap_rows_removed_from_target": overlap_removed,
            "target_usage": "Unlabeled scoring only; labels not loaded by the detector",
        },
        "seed": seed,
        "model": {
            "name": "marginal_shift_detector_comparison",
            "methods": list(settings["comparison_methods"]),
            "primary_method": settings["primary_method"],
        },
        "feature_set": {
            "method": "common_conservative_netflow_v2",
            "count": len(feature_names),
            "names": feature_names,
        },
        "metrics": {
            "macro_f1": None,
            "mcc": None,
            "false_positive_rate": None,
            "brier_score": None,
            "not_applicable_reason": "This unsupervised shift-detector comparison does not emit class probabilities or attack labels.",
        },
        "domain_shift_validation": {
            "window_size": int(settings["window_size"]),
            "quantile_bins": int(settings["quantile_bins"]),
            "calibration_alpha": float(settings["calibration_alpha"]),
            "mixture_fractions": [float(value) for value in settings["mixture_fractions"]],
            "mixture_windows_per_fraction": int(settings["mixture_windows_per_fraction"]),
            "source_roles": {name: int(len(index)) for name, index in roles.items()},
            "methods": methods,
            "success_criterion": {
                "primary_method": settings["primary_method"],
                "maximum_source_audit_false_alarm_rate": float(settings["maximum_source_audit_false_alarm_rate"]),
                "minimum_target_full_alarm_rate": 0.80,
                "minimum_half_target_mixture_alarm_rate": 0.80,
                "passed": primary_passed,
            },
            "scope_note": "Comparison detects covariate mismatch only; it neither classifies attacks nor establishes deployment remediation.",
        },
        "efficiency": {
            "serialized_model_bytes": 0,
            "single_record_latency_ms_median": None,
            "single_record_latency_ms_p95": None,
            "measurement": "Detector-comparison scoring duration is reported at experiment level; no classifier artifact is serialized.",
        },
        "durations": {
            "data_load_seconds": load_seconds,
            "all_methods_fit_and_score_seconds": duration,
        },
        "hardware": hardware_metadata(),
        "notes": [
            "The global canonical cap, source split, source validation roles, target cap, and mixture windows are label-blind.",
            "JSD, PSI, and KS use identical source reference, calibration, audit, target, and mixture windows within each seed.",
            "Target labels are not accessed for detector fitting, calibration, scoring, or threshold selection; recorded class counts are provenance only.",
            "The primary JSD method and criterion were frozen in configuration before this runner is executed.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/netflow_v2_label_blind.yaml")
    parser.add_argument("--source", choices=DATASET_KEYS, required=True)
    parser.add_argument("--target", choices=DATASET_KEYS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.source, args.target, args.seed, args.force)
    primary = result["domain_shift_validation"]["methods"][
        result["domain_shift_validation"]["success_criterion"]["primary_method"]
    ]
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "source_audit_false_alarm_rate": primary["source_audit"]["alarm_rate"],
        "target_alarm_rate": primary["target_full"]["alarm_rate"],
        "half_target_mixture_alarm_rate": primary["mixtures"]["0.5"]["alarm_rate"],
        "passed": result["domain_shift_validation"]["success_criterion"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
