"""Evaluate Benign-Envelope Fusion (BEF), an inexpensive unseen-campaign detector."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.research_common import (
    file_sha256,
    git_state,
    hardware_metadata,
    load_chronological_partitions,
    load_random_partitions,
    save_result,
    utc_now,
)
from experiments.run_research_proposed import _validation_roles
from src.evaluation.efficiency import benchmark_model
from src.evaluation.research_metrics import binary_metrics, select_thresholds
from src.features.stability import campaign_stable_feature_ranking
from src.models.lightweight import BenignEnvelopeFusionModel, build_lightweight_model


def _benign_profile(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    benign = X[y == 0]
    median = np.median(benign, axis=0)
    lower, upper = np.percentile(benign, [25, 75], axis=0)
    scale = upper - lower
    fallback = np.std(benign, axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return median.astype(np.float64), scale.astype(np.float64)


def _novelty(X: np.ndarray, median: np.ndarray, scale: np.ndarray, top_n: int = 3) -> np.ndarray:
    deviation = np.abs((X - median) / scale)
    count = min(top_n, deviation.shape[1])
    largest = np.partition(deviation, deviation.shape[1] - count, axis=1)[:, -count:]
    return np.log1p(largest.mean(axis=1))


def run(config: dict, seed: int, protocol: str = "chronological", force: bool = False) -> dict:
    output_dir = Path(config["output_dir"])
    base_name = config["primary_model"]
    experiment_id = f"bef_{protocol}_{base_name}_seed{seed}"
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

    fit_start = time.perf_counter()
    ranking = campaign_stable_feature_ranking(
        X_train,
        y_train,
        environments,
        feature_names,
        seed=seed,
        repeats=max(1, min(2, int(config.get("selection_repeats", 2)))),
        sample_rows_per_environment=max(5000, int(config.get("selection_sample_rows", 100000)) // 4),
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

    candidates = {}
    validation_by_budget = {}
    for budget in budgets:
        indices = ranked_indices[:budget]
        base = build_lightweight_model(base_name, seed)
        base.fit(X_train[:, indices], y_train)
        median, scale = _benign_profile(X_train[:, indices], y_train)
        calibration_indices = roles["calibration"]
        calibration_X = X_validation[calibration_indices][:, indices]
        fusion_X = np.column_stack(
            [
                base.predict_proba(calibration_X)[:, 1],
                _novelty(calibration_X, median, scale),
            ]
        )
        fusion = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=200, random_state=seed
            ),
        )
        fusion.fit(fusion_X, y_validation[calibration_indices])
        wrapped = BenignEnvelopeFusionModel(
            base, fusion, indices, [feature_names[i] for i in indices], median, scale
        )
        selection_probabilities = wrapped.predict_proba(X_validation[roles["feature_selection"]])[:, 1]
        selection_metrics = binary_metrics(
            y_validation[roles["feature_selection"]], selection_probabilities, 0.5
        )
        candidates[budget] = wrapped
        validation_by_budget[str(budget)] = {
            "macro_f1": selection_metrics["macro_f1"],
            "mcc": selection_metrics["mcc"],
            "false_positive_rate": selection_metrics["false_positive_rate"],
        }

    full_reference = validation_by_budget[str(len(feature_names))]["macro_f1"]
    tolerance = float(config.get("validation_tolerance_macro_f1", 0.01))
    eligible = [
        budget
        for budget in budgets
        if validation_by_budget[str(budget)]["macro_f1"] >= full_reference - tolerance
    ]
    selected_budget = min(eligible) if eligible else len(feature_names)
    model = candidates[selected_budget]
    fit_seconds = time.perf_counter() - fit_start

    threshold_probabilities = model.predict_proba(X_validation[roles["threshold"]])[:, 1]
    thresholds = select_thresholds(
        y_validation[roles["threshold"]],
        threshold_probabilities,
        config.get("fpr_constraint", 0.01),
    )
    test_probabilities = model.predict_proba(X_test)[:, 1]
    supervised_test_probabilities = model.supervised_estimator.predict_proba(
        X_test[:, model.feature_indices]
    )[:, 1]
    operating_points = {
        "supervised_selected_default": binary_metrics(y_test, supervised_test_probabilities, 0.5),
        "fusion_default": binary_metrics(y_test, test_probabilities, 0.5),
        "fusion_validation_macro_f1": binary_metrics(
            y_test, test_probabilities, thresholds["validation_macro_f1"]
        ),
        "fusion_fpr_constrained": binary_metrics(
            y_test, test_probabilities, thresholds["validation_fpr_constrained"]
        ),
    }
    primary_metrics = operating_points["fusion_fpr_constrained"]
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_model(model, X_test, artifact_path)
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
            "sampling_seed": seed,
            "day_assignment": (
                None
                if protocol == "random"
                else {"train": "Monday-Wednesday", "validation": "Thursday", "test": "Friday"}
            ),
            "validation_roles": {name: int(len(value)) for name, value in roles.items()},
            "test_usage": "Final evaluation only",
        },
        "seed": seed,
        "model": {
            "name": "benign_envelope_fusion",
            "supervised_model": base_name,
            "fusion_inputs": ["supervised_probability", "top3_robust_benign_deviation"],
            "threshold_policy": "validation_fpr_constrained",
            "fpr_constraint": float(config.get("fpr_constraint", 0.01)),
        },
        "feature_set": {
            "method": "campaign_stable_extra_trees",
            "count": selected_budget,
            "names": model.feature_names,
            "validation_tolerance_macro_f1": tolerance,
            "validation_by_budget": validation_by_budget,
            "ranking": json.loads(ranking.to_json(orient="records")),
        },
        "metrics": primary_metrics,
        "operating_points": operating_points,
        "validation_thresholds": thresholds,
        "efficiency": efficiency,
        "durations": {"data_load_seconds": load_seconds, "complete_fit_seconds": fit_seconds},
        "hardware": hardware_metadata(),
        "notes": [
            "BEF is an exploratory method adjustment motivated by baseline shift failure.",
            "Benign median and scale are fitted on training benign rows only.",
            "Fusion is fitted on the calibration validation role; budget and threshold use disjoint roles.",
            "Chronological CICIDS2017 results are compound day/campaign shift, not pure concept drift.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_baseline.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", choices=["random", "chronological"], default="chronological")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.seed, args.protocol, args.force)
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "selected_features": result["feature_set"]["count"],
                "macro_f1": result["metrics"]["macro_f1"],
                "mcc": result["metrics"]["mcc"],
                "fpr": result["metrics"]["false_positive_rate"],
                "brier": result["metrics"]["brier_score"],
                "model_bytes": result["efficiency"]["serialized_model_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

