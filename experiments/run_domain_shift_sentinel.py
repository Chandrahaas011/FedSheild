"""Run one source-calibrated NetFlow-v2 domain-shift sentinel experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.domain_shift_common import (
    benchmark_shift_guard,
    evaluate_guarded_stream,
    score_window_stream,
    source_audit_summary,
)
from experiments.research_common import git_state, hardware_metadata, save_result, utc_now
from experiments.run_cross_netflow import DATASET_KEYS, load_cross_partitions, signed_log1p
from experiments.uncertainty_common import primary_binary_metrics
from src.evaluation.abstention import fit_conformal_thresholds
from src.models.abstention import AbstainingClassifier
from src.models.domain_shift import QuantileJSDShiftSentinel, ShiftGuardedClassifier
from src.models.lightweight import build_lightweight_model


def _build_sentinel(settings: dict, seed: int, *, bins=None, aggregation="mean"):
    return QuantileJSDShiftSentinel(
        window_size=int(settings["window_size"]),
        quantile_bins=int(bins if bins is not None else settings["quantile_bins"]),
        calibration_alpha=float(settings["calibration_alpha"]),
        aggregation=aggregation,
        maximum_reference_rows=int(settings["maximum_reference_rows"]),
        minimum_calibration_windows=int(settings["minimum_calibration_windows"]),
        seed=seed,
    )


def _evaluate_sentinel(sentinel, X_test, y_test, seed: int) -> dict:
    benign = y_test == 0
    malicious = y_test == 1
    return {
        "source_audit": source_audit_summary(sentinel),
        "mixed_target": score_window_stream(sentinel, X_test, seed + 8101),
        "benign_target_diagnostic": score_window_stream(
            sentinel, X_test[benign], seed + 8111
        ),
        "malicious_target_diagnostic": score_window_stream(
            sentinel, X_test[malicious], seed + 8121
        ),
    }


def _trusted_audit_window(sentinel, X_validation: np.ndarray) -> np.ndarray:
    for score, index in zip(sentinel.audit_scores_, sentinel.audit_window_indices_):
        if score <= sentinel.threshold_:
            return X_validation[index]
    raise ValueError("no source-audit window passed the calibrated shift threshold")


def run(config: dict, source: str, target: str, seed: int, force: bool = False) -> dict:
    if source == target:
        raise ValueError("source and target must differ")
    experiment_id = f"domain_shift_{source}_to_{target}_hgb_seed{seed}"
    output_dir = Path(config["output_dir"])
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    settings = config["domain_shift_sentinel"]
    loading_config = dict(config)
    loading_config["target_cap_strategy"] = settings["target_cap_strategy"]
    metadata = json.loads(Path(config["metadata_path"]).read_text(encoding="utf-8"))
    feature_names = metadata["common_feature_columns"]

    load_start = time.perf_counter()
    parts, overlap_removed = load_cross_partitions(
        loading_config, source, target, seed
    )
    arrays = {
        role: (
            signed_log1p(frame[feature_names].to_numpy(dtype=np.float64)),
            frame["binary_label"].to_numpy(dtype=np.int8),
        )
        for role, frame in parts.items()
    }
    load_seconds = time.perf_counter() - load_start
    X_train, y_train = arrays["train"]
    X_validation, y_validation = arrays["validation"]
    X_test, y_test = arrays["test"]

    train_start = time.perf_counter()
    model = build_lightweight_model("hist_gradient_boosting", seed)
    model.fit(X_train, y_train)
    validation_probability = model.predict_proba(X_validation)[:, 1]
    conformal_thresholds = fit_conformal_thresholds(
        y_validation,
        validation_probability,
        alpha=float(config["uncertainty_alpha"]),
        class_conditional=True,
    )
    abstaining_model = AbstainingClassifier(
        model, "conformal", conformal_thresholds
    )
    sentinel = _build_sentinel(settings, seed).fit(X_train).calibrate(X_validation)
    training_seconds = time.perf_counter() - train_start

    target_probability = model.predict_proba(X_test)[:, 1]
    evaluations = _evaluate_sentinel(sentinel, X_test, y_test, seed)
    guarded_model = ShiftGuardedClassifier(abstaining_model, sentinel)
    guarded_outputs = evaluate_guarded_stream(
        guarded_model, X_test, y_test, seed + 8101
    )

    ablations = {}
    for specification in settings.get("ablations", []):
        ablation = _build_sentinel(
            settings,
            seed,
            bins=int(specification["quantile_bins"]),
            aggregation=str(specification["aggregation"]),
        ).fit(X_train).calibrate(X_validation)
        ablations[specification["name"]] = {
            "quantile_bins": int(specification["quantile_bins"]),
            "aggregation": specification["aggregation"],
            "threshold": float(ablation.threshold_),
            **_evaluate_sentinel(ablation, X_test, y_test, seed),
        }

    criteria = {
        "maximum_source_audit_false_alarm_rate": float(
            settings["maximum_source_audit_false_alarm_rate"]
        ),
        "minimum_mixed_target_detection_rate": float(
            settings["minimum_mixed_target_detection_rate"]
        ),
        "minimum_benign_target_detection_rate": float(
            settings["minimum_benign_target_detection_rate"]
        ),
    }
    primary_passed = bool(
        evaluations["source_audit"]["alarm_rate"]
        <= criteria["maximum_source_audit_false_alarm_rate"]
        and evaluations["mixed_target"]["alarm_rate"]
        >= criteria["minimum_mixed_target_detection_rate"]
        and evaluations["benign_target_diagnostic"]["alarm_rate"]
        >= criteria["minimum_benign_target_detection_rate"]
    )

    benchmark_window = _trusted_audit_window(sentinel, X_validation)
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_shift_guard(
        guarded_model, benchmark_window, artifact_path
    )
    # Compatibility keys keep the master result schema and generic resource tooling
    # machine-readable while retaining the correct batch-level units above.
    efficiency.update(
        {
            "serialized_model_bytes": efficiency["serialized_guarded_model_bytes"],
            "single_record_latency_ms_median": (
                efficiency["window_latency_ms_median"] / sentinel.window_size
            ),
            "single_record_latency_ms_p95": (
                efficiency["window_latency_ms_p95"] / sentinel.window_size
            ),
        }
    )

    source_meta = metadata["datasets"][source]
    target_meta = metadata["datasets"][target]
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": "UQ NetFlow-v2 guarded cross-dataset",
            "source": source,
            "target": target,
            "raw_archive_sha256": {
                "source": source_meta["archive_sha256"],
                "target": target_meta["archive_sha256"],
            },
            "rows": {name: int(len(frame)) for name, frame in parts.items()},
            "class_counts": {
                name: {
                    str(key): int(value)
                    for key, value in frame["binary_label"].value_counts().items()
                }
                for name, frame in parts.items()
            },
        },
        "protocol": {
            "name": "source_calibrated_domain_shift_sentinel",
            "base_protocol": "cross_dataset",
            "source": source,
            "target": target,
            "sampling_seed": seed,
            "cross_dataset_overlap_rows_removed_from_target": overlap_removed,
            "target_cap_strategy": settings["target_cap_strategy"],
            "target_usage": "Scored only after source-only method and threshold freeze",
            "target_label_usage": "Final classifier metrics and benign/malicious diagnostics only",
        },
        "seed": seed,
        "model": {
            "name": "shift_guarded_hgb",
            "base_estimator": "hist_gradient_boosting",
            "flow_rejection": "source_calibrated_mondrian_split_conformal",
            "deployment_sentinel": "source_quantile_histogram_mean_jsd",
        },
        "feature_set": {
            "method": "common_conservative_netflow_v2",
            "count": len(feature_names),
            "names": feature_names,
        },
        "metrics": primary_binary_metrics(y_test, target_probability),
        "source_validation_metrics": primary_binary_metrics(
            y_validation, validation_probability
        ),
        "domain_shift": {
            "primary": {
                "window_size": sentinel.window_size,
                "quantile_bins": sentinel.quantile_bins,
                "aggregation": sentinel.aggregation,
                "calibration_alpha": sentinel.calibration_alpha,
                "threshold": float(sentinel.threshold_),
                "reference_rows": sentinel.reference_rows_,
                "calibration_windows": int(len(sentinel.calibration_scores_)),
                "source_validation_rows_omitted": sentinel.validation_rows_omitted_,
                **evaluations,
            },
            "guarded_operational_outputs": guarded_outputs,
            "ablations": ablations,
            "success_criterion": {**criteria, "passed": primary_passed},
            "scope_note": (
                "The sentinel detects batch-level domain mismatch. It neither labels "
                "the cause as malicious nor repairs target classifier performance."
            ),
        },
        "efficiency": efficiency,
        "durations": {
            "data_load_seconds": load_seconds,
            "model_sentinel_training_seconds": training_seconds,
        },
        "hardware": hardware_metadata(),
        "notes": [
            "Feature bins, reference histograms, and the alarm threshold use source data only.",
            "Source calibration and audit windows are disjoint and label-blind.",
            "Target capping and mixed-window construction do not consult target labels.",
            "Benign-only and malicious-only target windows are retrospective diagnostics.",
            "DOMAIN_SHIFT_DETECTED is not counted as an attack detection.",
            "Canonical source files were previously capped per binary class and do not preserve native prevalence.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/netflow_v2_cross.yaml")
    parser.add_argument("--source", choices=DATASET_KEYS, required=True)
    parser.add_argument("--target", choices=DATASET_KEYS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.source, args.target, args.seed, args.force)
    primary = result["domain_shift"]["primary"]
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "source_audit_false_alarm_rate": primary["source_audit"]["alarm_rate"],
                "mixed_target_detection_rate": primary["mixed_target"]["alarm_rate"],
                "benign_target_detection_rate": primary["benign_target_diagnostic"]["alarm_rate"],
                "passed": result["domain_shift"]["success_criterion"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
