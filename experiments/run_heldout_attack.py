"""Run a classifier with one attack family excluded from all development data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.research_common import (
    file_sha256,
    git_state,
    hardware_metadata,
    load_heldout_attack_partitions,
    save_result,
    utc_now,
)
from src.evaluation.efficiency import benchmark_model
from src.evaluation.research_metrics import binary_metrics, select_thresholds
from src.models.lightweight import build_lightweight_model


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def run(config: dict, model_name: str, family: str, seed: int, force: bool = False) -> dict:
    output_dir = Path(config["output_dir"])
    experiment_id = f"heldout_{_slug(family)}_{model_name}_seed{seed}"
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
    partitions = load_heldout_attack_partitions(
        Path(config["dataset_path"]), feature_names, family, caps, seed
    )
    load_seconds = time.perf_counter() - load_start
    train, validation, test = partitions["train"], partitions["validation"], partitions["test"]
    X_train = train[feature_names].to_numpy(dtype=np.float32)
    y_train = train["binary_label"].to_numpy(dtype=np.int8)
    X_validation = validation[feature_names].to_numpy(dtype=np.float32)
    y_validation = validation["binary_label"].to_numpy(dtype=np.int8)
    X_test = test[feature_names].to_numpy(dtype=np.float32)
    y_test = test["binary_label"].to_numpy(dtype=np.int8)

    model = build_lightweight_model(model_name, seed)
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - train_start
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    thresholds = select_thresholds(
        y_validation, validation_probabilities, config.get("fpr_constraint", 0.01)
    )
    test_probabilities = model.predict_proba(X_test)[:, 1]
    operating_points = {
        name: binary_metrics(y_test, test_probabilities, threshold)
        for name, threshold in thresholds.items()
    }
    primary_metrics = operating_points["validation_fpr_constrained"]
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
            "test_family_counts": {
                str(k): int(v) for k, v in test["attack_class"].value_counts().items()
            },
        },
        "protocol": {
            "name": "heldout_attack_binary_detection",
            "heldout_attack_family": family,
            "development_exclusion": "Family removed from both training and validation",
            "test_composition": "Held-out family and benign only",
            "sampling_seed": seed,
            "threshold_selection": "Known-family validation data only",
        },
        "seed": seed,
        "model": {
            "name": model_name,
            "parameters": json.loads(json.dumps(model.get_params(deep=True), default=str)),
            "calibration": "none",
            "threshold_policy": "validation_fpr_constrained",
        },
        "feature_set": {"method": "all", "count": len(feature_names), "names": feature_names},
        "metrics": primary_metrics,
        "operating_points": operating_points,
        "validation_thresholds": thresholds,
        "efficiency": efficiency,
        "durations": {"data_load_seconds": load_seconds, "training_seconds": training_seconds},
        "hardware": hardware_metadata(),
        "notes": [
            "The classifier is binary and is not claimed to name the unseen family.",
            "Family labels are used only to construct the exclusion and report composition.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_baseline.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.model, args.family, args.seed, args.force)
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "macro_f1": result["metrics"]["macro_f1"],
                "malicious_recall": result["metrics"]["per_class"]["Malicious"]["recall"],
                "fpr": result["metrics"]["false_positive_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

