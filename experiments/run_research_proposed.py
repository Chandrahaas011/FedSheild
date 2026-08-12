"""Run the shift-aware compact IDS pipeline on the frozen random protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from experiments.research_common import (
    file_sha256,
    git_state,
    hardware_metadata,
    load_chronological_partitions,
    load_random_partitions,
    save_result,
    utc_now,
)
from src.evaluation.efficiency import benchmark_model
from src.evaluation.research_metrics import binary_metrics, select_thresholds
from src.features.stability import campaign_stable_feature_ranking
from src.models.lightweight import SelectedFeatureModel, build_lightweight_model


def _validation_roles(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels))
    selection, remainder = train_test_split(
        indices, train_size=0.40, random_state=seed, stratify=labels
    )
    calibration, threshold = train_test_split(
        remainder, test_size=0.50, random_state=seed + 1, stratify=labels[remainder]
    )
    return {
        "feature_selection": np.asarray(selection),
        "calibration": np.asarray(calibration),
        "threshold": np.asarray(threshold),
    }


def run(config: dict, seed: int, force: bool = False, protocol: str = "random") -> dict:
    output_dir = Path(config["output_dir"])
    model_name = config["primary_model"]
    experiment_id = f"proposed_{protocol}_campaign_stable_{model_name}_seed{seed}"
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    metadata_path = Path(config["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["feature_columns"]
    caps = {
        "train": config.get("max_train_rows"),
        "validation": config.get("max_validation_rows"),
        "test": config.get("max_test_rows"),
    }
    load_start = time.perf_counter()
    loader = load_random_partitions if protocol == "random" else load_chronological_partitions
    partitions = loader(Path(config["dataset_path"]), feature_names, caps, seed)
    load_seconds = time.perf_counter() - load_start
    train, validation, test = partitions["train"], partitions["validation"], partitions["test"]
    X_train = train[feature_names].to_numpy(dtype=np.float32)
    y_train = train["binary_label"].to_numpy(dtype=np.int8)
    environments = train["capture_day"].to_numpy()
    X_validation = validation[feature_names].to_numpy(dtype=np.float32)
    y_validation = validation["binary_label"].to_numpy(dtype=np.int8)
    X_test = test[feature_names].to_numpy(dtype=np.float32)
    y_test = test["binary_label"].to_numpy(dtype=np.int8)
    roles = _validation_roles(y_validation, seed)

    selection_start = time.perf_counter()
    ranking = campaign_stable_feature_ranking(
        X_train,
        y_train,
        environments,
        feature_names,
        seed=seed,
        repeats=max(1, min(3, int(config.get("selection_repeats", 2)))),
        sample_rows_per_environment=max(
            5000, int(config.get("selection_sample_rows", 100000)) // 4
        ),
    )
    ranked_names = ranking["feature"].tolist()
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    ranked_indices = [name_to_index[name] for name in ranked_names]

    budgets = sorted(
        {min(len(feature_names), int(value)) for value in config["feature_budgets"]},
        reverse=True,
    )
    if len(feature_names) not in budgets:
        budgets.insert(0, len(feature_names))
    validation_by_budget = {}
    fitted_by_budget = {}
    for budget in budgets:
        indices = ranked_indices[:budget]
        model = build_lightweight_model(model_name, seed)
        fit_start = time.perf_counter()
        model.fit(X_train[:, indices], y_train)
        fit_seconds = time.perf_counter() - fit_start
        probabilities = model.predict_proba(X_validation[roles["feature_selection"]][:, indices])[:, 1]
        metrics = binary_metrics(
            y_validation[roles["feature_selection"]], probabilities, threshold=0.5
        )
        validation_by_budget[str(budget)] = {
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
            "false_positive_rate": metrics["false_positive_rate"],
            "fit_seconds": fit_seconds,
        }
        fitted_by_budget[budget] = model

    full_reference = validation_by_budget[str(len(feature_names))]["macro_f1"]
    tolerance = float(config.get("validation_tolerance_macro_f1", 0.01))
    eligible = [
        budget
        for budget in budgets
        if validation_by_budget[str(budget)]["macro_f1"] >= full_reference - tolerance
    ]
    selected_budget = min(eligible) if eligible else len(feature_names)
    selected_indices = ranked_indices[:selected_budget]
    selected_names = [feature_names[index] for index in selected_indices]
    selected_model = fitted_by_budget[selected_budget]
    full_model = fitted_by_budget[len(feature_names)]
    selection_seconds = time.perf_counter() - selection_start

    calibration_start = time.perf_counter()
    calibrated = CalibratedClassifierCV(FrozenEstimator(selected_model), method="sigmoid")
    calibrated.fit(
        X_validation[roles["calibration"]][:, selected_indices],
        y_validation[roles["calibration"]],
    )
    calibration_seconds = time.perf_counter() - calibration_start

    threshold_y = y_validation[roles["threshold"]]
    threshold_uncalibrated = selected_model.predict_proba(
        X_validation[roles["threshold"]][:, selected_indices]
    )[:, 1]
    threshold_calibrated = calibrated.predict_proba(
        X_validation[roles["threshold"]][:, selected_indices]
    )[:, 1]
    thresholds = select_thresholds(
        threshold_y, threshold_calibrated, config.get("fpr_constraint", 0.01)
    )

    test_selected_uncalibrated = selected_model.predict_proba(X_test[:, selected_indices])[:, 1]
    test_selected_calibrated = calibrated.predict_proba(X_test[:, selected_indices])[:, 1]
    full_indices = ranked_indices[: len(feature_names)]
    test_full_uncalibrated = full_model.predict_proba(X_test[:, full_indices])[:, 1]
    ablations = {
        "full_features_uncalibrated_default": binary_metrics(y_test, test_full_uncalibrated, 0.5),
        "selected_uncalibrated_default": binary_metrics(y_test, test_selected_uncalibrated, 0.5),
        "selected_calibrated_default": binary_metrics(y_test, test_selected_calibrated, 0.5),
        "selected_calibrated_validation_macro_f1": binary_metrics(
            y_test, test_selected_calibrated, thresholds["validation_macro_f1"]
        ),
        "selected_calibrated_fpr_constrained": binary_metrics(
            y_test, test_selected_calibrated, thresholds["validation_fpr_constrained"]
        ),
    }
    primary_metrics = ablations["selected_calibrated_fpr_constrained"]

    wrapped_model = SelectedFeatureModel(calibrated, selected_indices, selected_names)
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_model(wrapped_model, X_test, artifact_path)
    ranking_records = json.loads(ranking.to_json(orient="records"))
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": metadata["dataset"],
            "raw_archive_sha256": metadata["raw_archive_sha256"],
            "metadata_sha256": file_sha256(metadata_path),
            "rows": {name: int(len(frame)) for name, frame in partitions.items()},
            "class_counts": {
                name: {str(k): int(v) for k, v in frame["binary_label"].value_counts().items()}
                for name, frame in partitions.items()
            },
        },
        "protocol": {
            "name": (
                "grouped_stratified_random"
                if protocol == "random"
                else "compound_chronological_campaign_shift"
            ),
            "split_seed": 42 if protocol == "random" else None,
            "sampling_seed": seed,
            "day_assignment": (
                None
                if protocol == "random"
                else {"train": "Monday-Wednesday", "validation": "Thursday", "test": "Friday"}
            ),
            "validation_roles": {name: int(len(indices)) for name, indices in roles.items()},
            "feature_selection_data": "Training environments plus feature-selection validation role",
            "calibration_data": "Disjoint calibration validation role",
            "threshold_data": "Disjoint threshold validation role",
            "test_usage": "Final evaluation only",
        },
        "seed": seed,
        "model": {
            "name": model_name,
            "parameters": json.loads(json.dumps(selected_model.get_params(deep=True), default=str)),
            "class_weighting": "balanced",
            "calibration": "sigmoid",
            "threshold_policy": "validation_fpr_constrained",
            "fpr_constraint": float(config.get("fpr_constraint", 0.01)),
        },
        "feature_set": {
            "method": "campaign_stable_extra_trees",
            "count": selected_budget,
            "names": selected_names,
            "validation_tolerance_macro_f1": tolerance,
            "validation_by_budget": validation_by_budget,
            "ranking": ranking_records,
        },
        "metrics": primary_metrics,
        "operating_points": ablations,
        "validation_thresholds": thresholds,
        "validation_calibration": {
            "uncalibrated": binary_metrics(threshold_y, threshold_uncalibrated, 0.5),
            "sigmoid": binary_metrics(threshold_y, threshold_calibrated, 0.5),
        },
        "efficiency": efficiency,
        "durations": {
            "data_load_seconds": load_seconds,
            "feature_selection_and_budget_search_seconds": selection_seconds,
            "calibration_seconds": calibration_seconds,
        },
        "hardware": hardware_metadata(),
        "notes": [
            "Binary benign-versus-malicious primary task.",
            "Campaign-stability ranking uses training rows only; test rows never influence ranking or budget.",
            "Capture day and attack family are confounded in CICIDS2017.",
            "Primary operating point is selected on validation data under the configured FPR constraint.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_baseline.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", choices=["random", "chronological"], default="random")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.seed, force=args.force, protocol=args.protocol)
    summary = {
        "experiment_id": result["experiment_id"],
        "selected_features": result["feature_set"]["count"],
        "macro_f1": result["metrics"]["macro_f1"],
        "mcc": result["metrics"]["mcc"],
        "fpr": result["metrics"]["false_positive_rate"],
        "brier": result["metrics"]["brier_score"],
        "model_bytes": result["efficiency"]["serialized_model_bytes"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
