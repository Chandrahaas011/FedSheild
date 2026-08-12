"""Run one frozen class-conditional robust-distance abstention experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.density_uncertainty_common import evaluate_density_uncertainty
from experiments.research_common import (
    file_sha256,
    git_state,
    hardware_metadata,
    save_result,
    utc_now,
)
from experiments.run_uncertainty import _load, _slug
from experiments.uncertainty_common import primary_binary_metrics, validation_roles
from src.evaluation.abstention import (
    fit_score_conformal_thresholds,
    fuse_candidate_scores,
    probability_nonconformity,
)
from src.evaluation.efficiency import benchmark_abstaining_model
from src.models.abstention import DistanceAwareAbstainingClassifier
from src.models.lightweight import build_lightweight_model
from src.models.robust_distance import ClassConditionalRobustDistance


def run(
    config: dict,
    dataset: str,
    protocol: str,
    seed: int,
    family: str | None = None,
    force: bool = False,
) -> dict:
    protocol_slug = protocol if protocol != "heldout" else f"heldout_{_slug(family or '')}"
    experiment_id = f"density_uncertainty_{dataset}_{protocol_slug}_hgb_seed{seed}"
    output_dir = Path(config["output_dir"])
    result_path = output_dir / f"{experiment_id}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    load_start = time.perf_counter()
    metadata, metadata_path, feature_names, parts = _load(
        config, dataset, protocol, family, seed
    )
    load_seconds = time.perf_counter() - load_start
    train, validation, test = parts["train"], parts["validation"], parts["test"]
    X_train = train[feature_names].to_numpy(dtype=np.float32)
    y_train = train["binary_label"].to_numpy(dtype=np.int8)
    X_validation = validation[feature_names].to_numpy(dtype=np.float32)
    y_validation = validation["binary_label"].to_numpy(dtype=np.int8)
    X_test = test[feature_names].to_numpy(dtype=np.float32)
    y_test = test["binary_label"].to_numpy(dtype=np.int8)
    roles = validation_roles(y_validation, seed)

    train_start = time.perf_counter()
    model = build_lightweight_model("hist_gradient_boosting", seed)
    model.fit(X_train, y_train)
    distance_model = ClassConditionalRobustDistance(
        reference_rows_per_class=int(config.get("density_reference_rows_per_class", 50_000)),
        seed=seed,
    ).fit(X_train, y_train)
    training_seconds = time.perf_counter() - train_start

    rejection_index = roles["rejection_calibration"]
    rejection_X = X_validation[rejection_index]
    rejection_y = y_validation[rejection_index]
    rejection_probability = model.predict_proba(rejection_X)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]
    rejection_novelty = distance_model.candidate_novelty(rejection_X)
    test_novelty = distance_model.candidate_novelty(X_test)
    alpha = float(config.get("uncertainty_alpha", 0.10))
    target_coverage = float(config.get("uncertainty_target_coverage", 0.90))
    probability_weight = float(config.get("density_fusion_weight", 0.50))
    uncertainty = evaluate_density_uncertainty(
        rejection_y,
        rejection_probability,
        rejection_novelty,
        y_test,
        test_probability,
        test_novelty,
        alpha,
        target_coverage,
        probability_weight,
    )

    fused_rejection_scores = fuse_candidate_scores(
        probability_nonconformity(rejection_probability),
        rejection_novelty,
        probability_weight,
    )
    fused_thresholds = fit_score_conformal_thresholds(
        rejection_y, fused_rejection_scores, alpha, class_conditional=True
    )
    abstaining_model = DistanceAwareAbstainingClassifier(
        model, distance_model, fused_thresholds, probability_weight
    )
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_abstaining_model(abstaining_model, X_test, artifact_path)

    raw_hash = (
        metadata["raw_archive_sha256"] if dataset == "cicids" else metadata["raw_sha256"]
    )
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": metadata["dataset"] if dataset == "cicids" else "RT-IoT2022",
            "raw_archive_sha256": raw_hash,
            "metadata_sha256": file_sha256(metadata_path),
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
            "name": f"density_uncertainty_{dataset}_{protocol}",
            "base_protocol": protocol,
            "heldout_attack_family": family,
            "sampling_seed": seed,
            "validation_roles": {name: int(len(index)) for name, index in roles.items()},
            "test_usage": "Final evaluation only; fusion weight and alpha frozen in config",
        },
        "seed": seed,
        "model": {
            "name": "distance_aware_hgb",
            "base_estimator": "hist_gradient_boosting",
            "distance_estimator": "class_conditional_robust_diagonal_distance",
            "probability_weight": probability_weight,
            "primary_rejection": "fused_distance_probability_mondrian",
            "alpha": alpha,
        },
        "feature_set": {"method": "all", "count": len(feature_names), "names": feature_names},
        "metrics": primary_binary_metrics(y_test, test_probability),
        "uncertainty": uncertainty,
        "efficiency": efficiency,
        "durations": {
            "data_load_seconds": load_seconds,
            "model_and_distance_training_seconds": training_seconds,
        },
        "hardware": hardware_metadata(),
        "notes": [
            "Distance centres, scales, and reference distributions use training data only.",
            "The 0.50 fusion weight and alpha 0.10 were frozen before test evaluation.",
            "Unknown denotes escalation rather than a correctly identified attack class.",
            "No conformal coverage guarantee is asserted under distribution shift.",
        ],
    }
    save_result(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["cicids", "rt_iot2022"], required=True)
    parser.add_argument("--protocol", choices=["random", "chronological", "heldout"], required=True)
    parser.add_argument("--family")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(config, args.dataset, args.protocol, args.seed, args.family, args.force)
    primary = result["uncertainty"]["methods"][
        "fused_distance_probability_mondrian_alpha_10"
    ]
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "coverage": primary["coverage"],
        "confident_false_negative_rate": primary["confident_false_negative_rate"],
        "malicious_rejection_rate": primary["malicious_rejection_rate"],
        "benign_rejection_rate": primary["benign_rejection_rate"],
        "success": result["uncertainty"]["density_success_criterion"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
