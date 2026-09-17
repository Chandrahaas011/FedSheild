"""Cross-dataset evaluation on the shared UQ NetFlow-v2 feature schema."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from experiments.research_common import git_state, hardware_metadata, save_result, utc_now
from experiments.uncertainty_common import evaluate_uncertainty, primary_binary_metrics
from src.evaluation.abstention import conformal_metrics, fit_conformal_thresholds
from src.evaluation.efficiency import benchmark_abstaining_model
from src.models.abstention import AbstainingClassifier
from src.models.lightweight import build_lightweight_model


DATASET_KEYS = ("nf_unsw_nb15_v2", "nf_ton_iot_v2")


def signed_log1p(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    transformed = np.sign(X) * np.log1p(np.abs(X))
    transformed = transformed.astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("signed-log transformed features must be finite")
    return transformed


def _cap(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(
        np.arange(len(frame)),
        train_size=int(maximum),
        random_state=seed,
        stratify=frame["binary_label"].to_numpy(),
    )
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def _cap_label_blind(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    """Cap rows deterministically without consulting target labels."""
    if maximum is None or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(frame), int(maximum), replace=False))
    return frame.iloc[selected].reset_index(drop=True)


def load_cross_partitions(config: dict, source: str, target: str, seed: int):
    if source == target or source not in DATASET_KEYS or target not in DATASET_KEYS:
        raise ValueError("source and target must be different supported datasets")
    processed = Path(config["processed_dir"])
    source_frame = pd.read_parquet(processed / f"{source}.parquet")
    target_frame = pd.read_parquet(processed / f"{target}.parquet")
    source_hashes = set(source_frame["feature_fingerprint"].astype("uint64"))
    overlap = target_frame["feature_fingerprint"].astype("uint64").isin(source_hashes)
    overlap_removed = int(overlap.sum())
    target_frame = target_frame[~overlap].reset_index(drop=True)

    train_index, validation_index = train_test_split(
        np.arange(len(source_frame)),
        test_size=0.20,
        random_state=seed + 4201,
        stratify=source_frame["binary_label"].to_numpy(),
    )
    train = _cap(
        source_frame.iloc[np.sort(train_index)].reset_index(drop=True),
        config.get("maximum_train_rows"),
        seed + 4211,
    )
    validation = _cap(
        source_frame.iloc[np.sort(validation_index)].reset_index(drop=True),
        config.get("maximum_validation_rows"),
        seed + 4221,
    )
    target_cap_strategy = config.get("target_cap_strategy", "stratified_label")
    if target_cap_strategy == "stratified_label":
        test = _cap(target_frame, config.get("maximum_target_test_rows"), seed + 4231)
    elif target_cap_strategy == "label_blind_random":
        test = _cap_label_blind(
            target_frame, config.get("maximum_target_test_rows"), seed + 4231
        )
    else:
        raise ValueError(f"unsupported target_cap_strategy: {target_cap_strategy}")
    return {"train": train, "validation": validation, "test": test}, overlap_removed


def _family_uncertainty(
    test: pd.DataFrame,
    test_probability: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    output = {}
    malicious = test["binary_label"].to_numpy(dtype=np.int8) == 1
    families = test.loc[malicious, "family_label"].astype(str).unique()
    for family in sorted(families):
        mask = malicious & (test["family_label"].astype(str).to_numpy() == family)
        metrics = conformal_metrics(
            np.ones(int(mask.sum()), dtype=np.int8), test_probability[mask], thresholds
        )
        output[family] = {
            "rows": int(mask.sum()),
            "base_malicious_recall": float((test_probability[mask] >= 0.5).mean()),
            "mondrian_confident_false_negative_rate": metrics["confident_false_negative_rate"],
            "mondrian_malicious_rejection_rate": metrics["malicious_rejection_rate"],
            "mondrian_automatic_malicious_detection_rate": metrics["automatic_malicious_detection_rate"],
        }
    return output


def run(config: dict, source: str, target: str, model_name: str, seed: int, force=False):
    model_slug = "hgb" if model_name == "hist_gradient_boosting" else model_name
    experiment_id = f"cross_netflow_{source}_to_{target}_{model_slug}_seed{seed}"
    output_dir = Path(config["output_dir"])
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(Path(config["metadata_path"]).read_text(encoding="utf-8"))
    feature_names = metadata["common_feature_columns"]

    load_start = time.perf_counter()
    parts, overlap_removed = load_cross_partitions(config, source, target, seed)
    load_seconds = time.perf_counter() - load_start
    arrays = {
        role: (
            signed_log1p(frame[feature_names].to_numpy(dtype=np.float64)),
            frame["binary_label"].to_numpy(dtype=np.int8),
        )
        for role, frame in parts.items()
    }
    X_train, y_train = arrays["train"]
    X_validation, y_validation = arrays["validation"]
    X_test, y_test = arrays["test"]

    train_start = time.perf_counter()
    model = build_lightweight_model(model_name, seed)
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - train_start
    validation_probability = model.predict_proba(X_validation)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]
    alpha = float(config["uncertainty_alpha"])
    uncertainty = evaluate_uncertainty(
        y_validation,
        validation_probability,
        y_test,
        test_probability,
        alpha,
        float(config["uncertainty_target_coverage"]),
    )
    thresholds = fit_conformal_thresholds(
        y_validation, validation_probability, alpha=alpha, class_conditional=True
    )
    wrapper = AbstainingClassifier(model, "conformal", thresholds)
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_abstaining_model(wrapper, X_test, artifact_path)

    source_meta = metadata["datasets"][source]
    target_meta = metadata["datasets"][target]
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": "UQ NetFlow-v2 cross-dataset",
            "source": source,
            "target": target,
            "raw_archive_sha256": {
                "source": source_meta["archive_sha256"],
                "target": target_meta["archive_sha256"],
            },
            "rows": {name: int(len(frame)) for name, frame in parts.items()},
            "class_counts": {
                name: {
                    str(k): int(v)
                    for k, v in frame["binary_label"].value_counts().items()
                }
                for name, frame in parts.items()
            },
        },
        "protocol": {
            "name": "cross_dataset_netflow_v2",
            "base_protocol": "cross_dataset",
            "source": source,
            "target": target,
            "sampling_seed": seed,
            "cross_dataset_overlap_rows_removed_from_target": overlap_removed,
            "target_usage": "Untouched final evaluation only",
            "threshold_source": "Source validation only",
        },
        "seed": seed,
        "model": {
            "name": model_slug,
            "base_estimator": model_name,
            "feature_transform": "signed_log1p",
            "primary_rejection": "source_calibrated_mondrian_split_conformal",
            "alpha": alpha,
        },
        "feature_set": {
            "method": "common_conservative_netflow_v2",
            "count": len(feature_names),
            "names": feature_names,
        },
        "metrics": primary_binary_metrics(y_test, test_probability),
        "source_validation_metrics": primary_binary_metrics(
            y_validation, validation_probability
        ),
        "uncertainty": uncertainty,
        "target_family_uncertainty": _family_uncertainty(
            parts["test"], test_probability, thresholds
        ),
        "efficiency": efficiency,
        "durations": {
            "data_load_seconds": load_seconds,
            "training_seconds": training_seconds,
        },
        "hardware": hardware_metadata(),
        "notes": [
            "Source and target use the same official 38-feature conservative NetFlow-v2 schema.",
            "A fixed signed-log1p transform controls heavy-tailed values without target fitting.",
            "Target labels select no feature, model, probability calibration, or threshold.",
            "Cross-dataset conformal coverage is empirical; exchangeability is not assumed.",
            "Canonical data are bounded hash-priority unique-pattern samples.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/netflow_v2_cross.yaml")
    parser.add_argument("--source", choices=DATASET_KEYS, required=True)
    parser.add_argument("--target", choices=DATASET_KEYS, required=True)
    parser.add_argument("--model", choices=["decision_tree", "hist_gradient_boosting"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.source, args.target, args.model, args.seed, args.force)
    mondrian = result["uncertainty"]["methods"]["mondrian_conformal_alpha_10"]
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "macro_f1": result["metrics"]["macro_f1"],
        "malicious_recall": result["metrics"]["per_class"]["Malicious"]["recall"],
        "coverage": mondrian["coverage"],
        "confident_false_negative_rate": mondrian["confident_false_negative_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
