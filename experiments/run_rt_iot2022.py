"""Run RT-IoT2022 random-split or held-out-family replication experiments."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.research_common import file_sha256, git_state, hardware_metadata, save_result, utc_now
from src.evaluation.efficiency import benchmark_model
from src.evaluation.research_metrics import binary_metrics, select_thresholds
from src.models.lightweight import build_lightweight_model


def _partitions(config: dict, metadata: dict, heldout: str | None) -> dict[str, pd.DataFrame]:
    columns = metadata["feature_columns"] + ["attack_family", "binary_label", "feature_fingerprint", "split_random"]
    frame = pd.read_parquet(config["dataset_path"], columns=columns)
    output = {}
    for name in ("train", "validation", "test"):
        part = frame[frame["split_random"] == name].copy()
        if heldout:
            if name == "test":
                part = part[(part["binary_label"] == 0) | (part["attack_family"] == heldout)]
            else:
                part = part[part["attack_family"] != heldout]
        output[name] = part.reset_index(drop=True)
    if heldout and output["test"]["binary_label"].nunique() != 2:
        raise ValueError(f"Held-out test for {heldout} lacks both classes")
    return output


def run(config: dict, model_name: str, seed: int, heldout: str | None = None, force: bool = False) -> dict:
    protocol_slug = f"heldout_{heldout.lower()}" if heldout else "random"
    experiment_id = f"rt_iot2022_{protocol_slug}_{model_name}_seed{seed}"
    output_dir = Path(config["output_dir"])
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    metadata_path = Path(config["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["feature_columns"]
    load_start = time.perf_counter()
    parts = _partitions(config, metadata, heldout)
    load_seconds = time.perf_counter() - load_start
    arrays = {}
    for name, part in parts.items():
        arrays[name] = (
            part[feature_names].to_numpy(dtype=np.float32),
            part["binary_label"].to_numpy(dtype=np.int8),
        )
    model = build_lightweight_model(model_name, seed)
    train_start = time.perf_counter()
    model.fit(*arrays["train"])
    train_seconds = time.perf_counter() - train_start
    val_probability = model.predict_proba(arrays["validation"][0])[:, 1]
    thresholds = select_thresholds(arrays["validation"][1], val_probability, config.get("fpr_constraint", 0.01))
    test_probability = model.predict_proba(arrays["test"][0])[:, 1]
    operating_points = {
        name: binary_metrics(arrays["test"][1], test_probability, threshold)
        for name, threshold in thresholds.items()
    }
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_model(model, arrays["test"][0], artifact_path)
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": "RT-IoT2022",
            "raw_archive_sha256": metadata["raw_sha256"],
            "metadata_sha256": file_sha256(metadata_path),
            "rows": {name: int(len(part)) for name, part in parts.items()},
            "class_counts": {name: {str(k): int(v) for k, v in part["binary_label"].value_counts().items()} for name, part in parts.items()},
        },
        "protocol": {
            "name": "rt_iot2022_heldout_attack_binary" if heldout else "rt_iot2022_grouped_stratified_random",
            "heldout_attack_family": heldout,
            "split_seed": 42,
            "sampling_seed": seed,
            "preprocessing": "Conservative numeric flow features; estimator transforms fit on training only",
            "threshold_selection": "Validation only; default threshold is primary",
        },
        "seed": seed,
        "model": {"name": model_name, "parameters": json.loads(json.dumps(model.get_params(deep=True), default=str)), "calibration": "none"},
        "feature_set": {"method": "conservative_numeric_flow_features", "count": len(feature_names), "names": feature_names},
        "metrics": operating_points["default"],
        "operating_points": operating_points,
        "validation_thresholds": thresholds,
        "efficiency": efficiency,
        "durations": {"data_load_seconds": load_seconds, "training_seconds": train_seconds},
        "hardware": hardware_metadata(),
        "notes": [
            "Independent IoT-dataset replication; no feature-space alignment with CICIDS2017 is claimed.",
            "Export index, endpoint ports, protocol, and service were conservatively excluded to reduce identifier artifacts.",
            "Held-out protocols remove the named attack from both training and validation.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rt_iot2022.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--heldout")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.model, args.seed, args.heldout, args.force)
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "macro_f1": result["metrics"]["macro_f1"],
        "malicious_recall": result["metrics"]["per_class"]["Malicious"]["recall"],
        "fpr": result["metrics"]["false_positive_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
