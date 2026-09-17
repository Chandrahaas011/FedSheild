"""Evaluate resource-bounded, validation-gated recovery after cross-testbed shift."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.domain_shift_common import label_blind_source_roles, label_blind_window_indices
from experiments.research_common import git_state, hardware_metadata, save_result, utc_now
from experiments.run_cross_netflow import DATASET_KEYS, signed_log1p
from experiments.run_domain_shift_validation import load_label_blind_partitions
from experiments.safe_resumption_common import (
    allocate_recovery_roles,
    attach_review_label_noise,
    choose_resumption_action,
    gate_passes,
    nested_balanced_review,
    uncertainty_aware_gate_passes,
)
from src.evaluation.efficiency import benchmark_model
from src.evaluation.research_metrics import binary_metrics
from src.models.domain_shift import MarginalShiftSentinel
from src.models.lightweight import build_lightweight_model
from src.models.recovery import ProbabilityRecalibrator


def _arrays(frame, feature_names):
    return (
        signed_log1p(frame[feature_names].to_numpy(dtype=np.float64)),
        frame["binary_label"].to_numpy(dtype=np.int8),
    )


def _diagnosis(
    X_source_train: np.ndarray,
    X_source_validation: np.ndarray,
    X_target_audit: np.ndarray,
    feature_names: list[str],
    settings: dict,
    recovery: dict,
    seed: int,
) -> dict:
    roles = label_blind_source_roles(
        len(X_source_validation), int(settings["window_size"]), seed
    )
    detector = MarginalShiftSentinel(
        method="jsd_mean",
        window_size=int(settings["window_size"]),
        quantile_bins=int(settings["quantile_bins"]),
        calibration_alpha=float(settings["calibration_alpha"]),
        maximum_reference_rows=int(settings["maximum_reference_rows"]),
        maximum_ks_reference_rows=int(settings["maximum_ks_reference_rows"]),
        minimum_calibration_windows=int(settings["minimum_calibration_windows"]),
        seed=seed,
    ).fit(X_source_train).calibrate(
        X_source_validation[roles["calibration"]],
        X_source_validation[roles["audit"]],
    )
    windows, tail = label_blind_window_indices(
        len(X_target_audit), detector.window_size, seed + 11201
    )
    windows = windows[: int(recovery["diagnosis_windows"])]
    score_matrix = np.vstack(
        [detector.feature_scores(X_target_audit[index]) for index in windows]
    )
    mean_scores = score_matrix.mean(axis=0)
    std_scores = score_matrix.std(axis=0, ddof=1) if len(windows) > 1 else np.zeros_like(mean_scores)
    order = np.argsort(-mean_scores)
    top_n = int(recovery["diagnosis_top_features"])
    ranked = [
        {
            "rank": rank + 1,
            "feature": feature_names[int(index)],
            "mean_jsd": float(mean_scores[index]),
            "std_jsd": float(std_scores[index]),
            "share_of_total_mean_jsd": float(mean_scores[index] / mean_scores.sum()),
        }
        for rank, index in enumerate(order[:top_n])
    ]
    return {
        "method": "mean per-feature JSD on label-blind final-audit windows",
        "windows": int(len(windows)),
        "incomplete_rows_omitted": int(tail),
        "threshold": float(detector.threshold_),
        "window_score_mean": float(score_matrix.mean(axis=1).mean()),
        "top_features": ranked,
        "all_feature_mean_jsd": {
            feature_names[index]: float(mean_scores[index]) for index in range(len(feature_names))
        },
    }


def _evaluate_action(
    action: str,
    label_budget: int,
    model,
    X_validation,
    y_validation,
    X_audit,
    y_audit,
    X_source_validation,
    y_source_validation,
    criteria,
    artifact_path: Path,
    training_seconds: float,
    gate_mode: str = "point",
    X_shadow_validation=None,
    y_shadow_validation=None,
) -> dict:
    efficiency = benchmark_model(model, X_audit, artifact_path)
    validation_metrics = binary_metrics(y_validation, model.predict_proba(X_validation)[:, 1])
    audit_metrics = binary_metrics(y_audit, model.predict_proba(X_audit)[:, 1])
    source_retention = binary_metrics(
        y_source_validation, model.predict_proba(X_source_validation)[:, 1]
    )
    shadow_metrics = None
    if gate_mode == "wilson_fpr_upper_95":
        passed, validation_fpr_upper = uncertainty_aware_gate_passes(
            validation_metrics, efficiency, criteria
        )
        audit_passed, audit_fpr_upper = uncertainty_aware_gate_passes(
            audit_metrics, efficiency, criteria
        )
    elif gate_mode == "point":
        passed = gate_passes(validation_metrics, efficiency, criteria)
        audit_passed = gate_passes(audit_metrics, efficiency, criteria)
        validation_fpr_upper = None
        audit_fpr_upper = None
    else:
        raise ValueError(f"Unsupported recovery gate_mode: {gate_mode}")
    shadow_passed = None
    shadow_fpr_upper = None
    if X_shadow_validation is not None:
        shadow_metrics = binary_metrics(
            y_shadow_validation,
            model.predict_proba(X_shadow_validation)[:, 1],
        )
        if gate_mode == "wilson_fpr_upper_95":
            shadow_passed, shadow_fpr_upper = uncertainty_aware_gate_passes(
                shadow_metrics, efficiency, criteria
            )
        else:
            shadow_passed = gate_passes(shadow_metrics, efficiency, criteria)
        passed = bool(passed and shadow_passed)
    return {
        "action": action,
        "label_budget": int(label_budget),
        "training_seconds": float(training_seconds),
        "validation_metrics": validation_metrics,
        "final_audit_metrics": audit_metrics,
        "source_retention_metrics": source_retention,
        "efficiency": efficiency,
        "gate_passed": passed,
        "final_audit_passed": audit_passed,
        "unsafe_resumption": bool(passed and not audit_passed),
        "gate_mode": gate_mode,
        "validation_fpr_wilson_upper_95": validation_fpr_upper,
        "final_audit_fpr_wilson_upper_95": audit_fpr_upper,
        "shadow_validation_metrics": shadow_metrics,
        "shadow_validation_passed": shadow_passed,
        "shadow_validation_fpr_wilson_upper_95": shadow_fpr_upper,
    }


def run(config: dict, source: str, target: str, seed: int, force: bool = False) -> dict:
    experiment_prefix = config.get("experiment_prefix", "safe_resumption")
    experiment_id = f"{experiment_prefix}_{source}_to_{target}_seed{seed}"
    output_dir = Path(config["output_dir"])
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    recovery = config["recovery"]
    criteria = recovery["gate"]
    gate_mode = recovery.get("gate_mode", "point")
    metadata = json.loads(Path(config["metadata_path"]).read_text(encoding="utf-8"))
    feature_names = metadata["common_feature_columns"]

    load_start = time.perf_counter()
    parts, overlap_removed = load_label_blind_partitions(config, source, target, seed)
    target_roles = allocate_recovery_roles(
        parts["target"],
        int(recovery["maximum_review_rows"]),
        int(recovery["validation_rows"]),
        int(recovery["final_audit_rows"]),
        seed,
        int(recovery.get("shadow_validation_rows", 0)),
    )
    X_source_train, y_source_train = _arrays(parts["train"], feature_names)
    X_source_validation, y_source_validation = _arrays(parts["validation"], feature_names)
    X_target_validation, y_target_validation = _arrays(target_roles["validation"], feature_names)
    X_target_audit, y_target_audit = _arrays(target_roles["audit"], feature_names)
    if "shadow_validation" in target_roles:
        X_shadow_validation, y_shadow_validation = _arrays(
            target_roles["shadow_validation"], feature_names
        )
    else:
        X_shadow_validation = y_shadow_validation = None
    load_seconds = time.perf_counter() - load_start

    source_start = time.perf_counter()
    source_model = build_lightweight_model(recovery["source_model"], seed)
    source_model.fit(X_source_train, y_source_train)
    source_training_seconds = time.perf_counter() - source_start
    artifact_dir = Path(config["artifact_dir"])
    actions = [
        _evaluate_action(
            "no_adaptation",
            0,
            source_model,
            X_target_validation,
            y_target_validation,
            X_target_audit,
            y_target_audit,
            X_source_validation,
            y_source_validation,
            criteria,
            artifact_dir / f"{experiment_id}_no_adaptation.joblib",
            source_training_seconds,
            gate_mode,
            X_shadow_validation,
            y_shadow_validation,
        )
    ]

    review_noise_rate = float(recovery.get("review_label_noise_rate", 0.0))
    configured_by_class = recovery.get("review_label_noise_by_class")
    review_noise_by_class = (
        {int(label): float(rate) for label, rate in configured_by_class.items()}
        if configured_by_class is not None else None
    )
    noisy_review_pool = attach_review_label_noise(
        target_roles["review"], review_noise_rate, seed, review_noise_by_class
    )
    for budget in recovery["label_budgets"]:
        review = nested_balanced_review(noisy_review_pool, int(budget), seed)
        X_review = signed_log1p(review[feature_names].to_numpy(dtype=np.float64))
        y_review = review["recovery_training_label"].to_numpy(dtype=np.int8)

        start = time.perf_counter()
        recalibrated = ProbabilityRecalibrator(source_model, seed).fit(X_review, y_review)
        duration = time.perf_counter() - start
        actions.append(
            _evaluate_action(
                "recalibration",
                int(budget),
                recalibrated,
                X_target_validation,
                y_target_validation,
                X_target_audit,
                y_target_audit,
                X_source_validation,
                y_source_validation,
                criteria,
                artifact_dir / f"{experiment_id}_recalibration_{budget}.joblib",
                duration,
                gate_mode,
                X_shadow_validation,
                y_shadow_validation,
            )
        )

        for specification in recovery.get("additional_retraining_models", []):
            action_name = specification["action"]
            start = time.perf_counter()
            robust_model = build_lightweight_model(
                specification["model"], seed + int(budget) + 17003
            )
            robust_model.fit(X_review, y_review)
            duration = time.perf_counter() - start
            actions.append(
                _evaluate_action(
                    action_name,
                    int(budget),
                    robust_model,
                    X_target_validation,
                    y_target_validation,
                    X_target_audit,
                    y_target_audit,
                    X_source_validation,
                    y_source_validation,
                    criteria,
                    artifact_dir / f"{experiment_id}_{action_name}_{budget}.joblib",
                    duration,
                    gate_mode,
                    X_shadow_validation,
                    y_shadow_validation,
                )
            )

        start = time.perf_counter()
        retrained = build_lightweight_model("decision_tree", seed + int(budget))
        retrained.fit(X_review, y_review)
        duration = time.perf_counter() - start
        actions.append(
            _evaluate_action(
                "compact_retraining",
                int(budget),
                retrained,
                X_target_validation,
                y_target_validation,
                X_target_audit,
                y_target_audit,
                X_source_validation,
                y_source_validation,
                criteria,
                artifact_dir / f"{experiment_id}_compact_retraining_{budget}.joblib",
                duration,
                gate_mode,
                X_shadow_validation,
                y_shadow_validation,
            )
        )

    selected = choose_resumption_action(actions)
    selected_summary = (
        {
            "decision": "SAFE_TO_RESUME",
            "action": selected["action"],
            "label_budget": selected["label_budget"],
            "gate_passed": True,
            "final_audit_passed": selected["final_audit_passed"],
            "unsafe_resumption": selected["unsafe_resumption"],
            "selection_basis": (
                "minimum label budget, then serialized size, then validation macro-F1; "
                "all configured validation gates required; final audit excluded"
            ),
        }
        if selected is not None
        else {
            "decision": "RECALIBRATION_REQUIRED",
            "action": None,
            "label_budget": None,
            "gate_passed": False,
            "final_audit_passed": None,
            "unsafe_resumption": False,
            "selection_basis": "no action met all frozen validation and resource criteria",
        }
    )
    diagnosis = _diagnosis(
        X_source_train,
        X_source_validation,
        X_target_audit,
        feature_names,
        config["domain_shift_validation"],
        recovery,
        seed,
    )
    baseline = actions[0]
    source_meta = metadata["datasets"][source]
    target_meta = metadata["datasets"][target]
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": "UQ NetFlow-v2 bounded safe-resumption study",
            "source": source,
            "target": target,
            "raw_archive_sha256": {
                "source": source_meta["archive_sha256"],
                "target": target_meta["archive_sha256"],
            },
            "rows": {
                "source_train": int(len(parts["train"])),
                "source_validation": int(len(parts["validation"])),
                **{f"target_{name}": int(len(frame)) for name, frame in target_roles.items()},
            },
            "class_counts": {
                f"target_{name}": {
                    str(label): int(count)
                    for label, count in frame["binary_label"].value_counts().items()
                }
                for name, frame in target_roles.items()
            },
        },
        "protocol": {
            "name": config.get(
                "protocol_name", "resource_bounded_validation_gated_safe_resumption"
            ),
            "base_protocol": "supervised_cross_dataset_recovery",
            "source": source,
            "target": target,
            "sampling_seed": seed,
            "cross_dataset_overlap_rows_removed_from_target": overlap_removed,
            "review_usage": "adaptation only; balanced nested labeled budgets",
            "validation_usage": "resumption gate and action selection only",
            "shadow_validation_usage": (
                "independent second resumption gate only; final audit excluded"
                if "shadow_validation" in target_roles else None
            ),
            "final_audit_usage": "reporting only; excluded from action and budget selection",
            "gate_mode": gate_mode,
            "review_label_noise_rate": review_noise_rate,
            "review_label_noise_by_class": review_noise_by_class,
            "review_label_noise_usage": "training review labels only; validation, shadow, and audit labels remain clean",
        },
        "seed": seed,
        "model": {
            "name": "safe_resumption_policy",
            "source_estimator": recovery["source_model"],
            "actions": recovery["actions"],
        },
        "feature_set": {
            "method": "common_conservative_netflow_v2",
            "count": len(feature_names),
            "names": feature_names,
        },
        "metrics": baseline["final_audit_metrics"],
        "efficiency": baseline["efficiency"],
        "durations": {
            "data_load_seconds": load_seconds,
            "source_training_seconds": source_training_seconds,
            "all_recovery_training_seconds": float(sum(action["training_seconds"] for action in actions[1:])),
        },
        "hardware": hardware_metadata(),
        "safe_resumption": {
            "gate_criteria": criteria,
            "gate_mode": gate_mode,
            "review_label_noise_rate": review_noise_rate,
            "review_label_noise_by_class": review_noise_by_class,
            "label_budgets": [int(value) for value in recovery["label_budgets"]],
            "actions": actions,
            "selected_policy": selected_summary,
            "diagnosis": diagnosis,
        },
        "notes": [
            "Target labels are used only after shift detection and within explicit recovery roles.",
            "Review, resumption-validation, and final-audit rows are disjoint.",
            "The selected action and label budget use validation and resource measurements only; final-audit metrics are reporting-only.",
            "SAFE_TO_RESUME is a protocol decision under frozen criteria, not a guarantee of production safety.",
            "Feature-level JSD is descriptive shift attribution, not causal attack explanation.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_resumption.yaml")
    parser.add_argument("--source", choices=DATASET_KEYS, required=True)
    parser.add_argument("--target", choices=DATASET_KEYS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.source, args.target, args.seed, args.force)
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        **result["safe_resumption"]["selected_policy"],
    }, indent=2))


if __name__ == "__main__":
    main()
