"""Run one leakage-separated confidence/conformal abstention experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from experiments.research_common import (
    file_sha256,
    git_state,
    hardware_metadata,
    load_chronological_partitions,
    load_heldout_attack_partitions,
    load_random_partitions,
    save_result,
    utc_now,
)
from experiments.run_rt_iot2022 import _partitions as load_rt_partitions
from experiments.uncertainty_common import (
    evaluate_uncertainty,
    primary_binary_metrics,
    validation_roles,
)
from src.evaluation.abstention import fit_conformal_thresholds
from src.evaluation.efficiency import benchmark_abstaining_model
from src.features.stability import campaign_stable_feature_ranking
from src.models.abstention import AbstainingClassifier
from src.models.lightweight import SelectedFeatureModel, build_lightweight_model


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def _load(config: dict, dataset: str, protocol: str, family: str | None, seed: int):
    metadata_path = Path(config["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata["feature_columns"]
    if dataset == "cicids":
        caps = {
            "train": config.get("max_train_rows"),
            "validation": config.get("max_validation_rows"),
            "test": config.get("max_test_rows"),
        }
        if protocol == "random":
            parts = load_random_partitions(Path(config["dataset_path"]), feature_names, caps, seed)
        elif protocol == "chronological":
            parts = load_chronological_partitions(Path(config["dataset_path"]), feature_names, caps, seed)
        elif protocol == "heldout":
            if not family:
                raise ValueError("heldout protocol requires --family")
            parts = load_heldout_attack_partitions(
                Path(config["dataset_path"]), feature_names, family, caps, seed
            )
        else:
            raise ValueError(f"unsupported CICIDS protocol: {protocol}")
    else:
        if protocol not in {"random", "heldout"}:
            raise ValueError("RT-IoT2022 supports random and heldout protocols")
        parts = load_rt_partitions(config, metadata, family if protocol == "heldout" else None)
    return metadata, metadata_path, feature_names, parts


def _fit_compact(
    config: dict,
    seed: int,
    feature_names: list[str],
    train,
    validation,
    roles: dict[str, np.ndarray],
):
    X_train = train[feature_names].to_numpy(dtype=np.float32)
    y_train = train["binary_label"].to_numpy(dtype=np.int8)
    environments = train["capture_day"].to_numpy()
    X_validation = validation[feature_names].to_numpy(dtype=np.float32)
    y_validation = validation["binary_label"].to_numpy(dtype=np.int8)
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
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    ranked_indices = [name_to_index[name] for name in ranking["feature"]]
    budgets = sorted(
        {min(len(feature_names), int(value)) for value in config["feature_budgets"]}
    )
    if len(feature_names) not in budgets:
        budgets.append(len(feature_names))
    selection_index = roles["model_selection"]
    fitted = {}
    validation_scores = {}
    from src.evaluation.research_metrics import binary_metrics

    for budget in budgets:
        indices = ranked_indices[:budget]
        model = build_lightweight_model(config["primary_model"], seed)
        model.fit(X_train[:, indices], y_train)
        probability = model.predict_proba(
            X_validation[selection_index][:, indices]
        )[:, 1]
        validation_scores[str(budget)] = binary_metrics(
            y_validation[selection_index], probability, 0.5
        )["macro_f1"]
        fitted[budget] = model
    full_score = validation_scores[str(len(feature_names))]
    tolerance = float(config.get("validation_tolerance_macro_f1", 0.01))
    eligible = [
        budget
        for budget in budgets
        if validation_scores[str(budget)] >= full_score - tolerance
    ]
    selected_budget = min(eligible) if eligible else len(feature_names)
    selected_indices = ranked_indices[:selected_budget]
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(fitted[selected_budget]), method="sigmoid"
    )
    calibration_index = roles["probability_calibration"]
    calibrated.fit(
        X_validation[calibration_index][:, selected_indices],
        y_validation[calibration_index],
    )
    selected_names = [feature_names[index] for index in selected_indices]
    wrapper = SelectedFeatureModel(calibrated, selected_indices, selected_names)
    return wrapper, {
        "method": "campaign_stable_extra_trees_then_sigmoid",
        "count": int(selected_budget),
        "names": selected_names,
        "validation_macro_f1_by_budget": validation_scores,
        "validation_tolerance_macro_f1": tolerance,
    }


def run(
    config: dict,
    dataset: str,
    protocol: str,
    seed: int,
    family: str | None = None,
    compact: bool = False,
    force: bool = False,
) -> dict:
    if compact and (dataset != "cicids" or protocol == "heldout"):
        raise ValueError("compact uncertainty ablation is limited to CICIDS random/chronological")
    protocol_slug = protocol if protocol != "heldout" else f"heldout_{_slug(family or '')}"
    model_slug = "compact_hgb" if compact else "hgb"
    experiment_id = f"uncertainty_{dataset}_{protocol_slug}_{model_slug}_seed{seed}"
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
    if compact:
        model, feature_set = _fit_compact(
            config, seed, feature_names, train, validation, roles
        )
    else:
        model = build_lightweight_model("hist_gradient_boosting", seed)
        model.fit(X_train, y_train)
        feature_set = {
            "method": "all" if dataset == "cicids" else "conservative_numeric_flow_features",
            "count": len(feature_names),
            "names": feature_names,
        }
    training_seconds = time.perf_counter() - train_start

    rejection_index = roles["rejection_calibration"]
    rejection_probability = model.predict_proba(X_validation[rejection_index])[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]
    alpha = float(config.get("uncertainty_alpha", 0.10))
    target_coverage = float(config.get("uncertainty_target_coverage", 0.90))
    uncertainty = evaluate_uncertainty(
        y_validation[rejection_index],
        rejection_probability,
        y_test,
        test_probability,
        alpha,
        target_coverage,
    )
    mondrian_thresholds = fit_conformal_thresholds(
        y_validation[rejection_index],
        rejection_probability,
        alpha=alpha,
        class_conditional=True,
    )
    abstaining_model = AbstainingClassifier(
        model, method="conformal", parameters=mondrian_thresholds
    )
    artifact_path = Path(config["artifact_dir"]) / f"{experiment_id}.joblib"
    efficiency = benchmark_abstaining_model(abstaining_model, X_test, artifact_path)

    if dataset == "cicids":
        raw_hash = metadata["raw_archive_sha256"]
        dataset_name = metadata["dataset"]
    else:
        raw_hash = metadata["raw_sha256"]
        dataset_name = "RT-IoT2022"
    protocol_name = f"uncertainty_{dataset}_{protocol}"
    result = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "complete",
        "code_version": git_state(),
        "dataset": {
            "name": dataset_name,
            "raw_archive_sha256": raw_hash,
            "metadata_sha256": file_sha256(metadata_path),
            "rows": {name: int(len(frame)) for name, frame in parts.items()},
            "class_counts": {
                name: {str(k): int(v) for k, v in frame["binary_label"].value_counts().items()}
                for name, frame in parts.items()
            },
        },
        "protocol": {
            "name": protocol_name,
            "base_protocol": protocol,
            "heldout_attack_family": family,
            "sampling_seed": seed,
            "validation_roles": {name: int(len(index)) for name, index in roles.items()},
            "validation_role_separation": (
                "40% model selection, 30% probability calibration, 30% rejection calibration"
            ),
            "test_usage": "Final evaluation only; alpha and target coverage frozen in config",
        },
        "seed": seed,
        "model": {
            "name": model_slug,
            "base_estimator": "hist_gradient_boosting",
            "probability_calibration": "sigmoid" if compact else "none",
            "primary_rejection": "mondrian_split_conformal",
            "alpha": alpha,
        },
        "feature_set": feature_set,
        "metrics": primary_binary_metrics(y_test, test_probability),
        "uncertainty": uncertainty,
        "efficiency": efficiency,
        "durations": {
            "data_load_seconds": load_seconds,
            "training_and_optional_selection_seconds": training_seconds,
        },
        "hardware": hardware_metadata(),
        "notes": [
            "Unknown is an abstention outcome, not a third learned attack class.",
            "Mondrian calibration uses only the disjoint rejection-calibration validation role.",
            "Conformal exchangeability guarantees are not asserted under distribution shift.",
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
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(
        config,
        args.dataset,
        args.protocol,
        args.seed,
        family=args.family,
        compact=args.compact,
        force=args.force,
    )
    primary = result["uncertainty"]["methods"]["mondrian_conformal_alpha_10"]
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "base_macro_f1": result["metrics"]["macro_f1"],
        "coverage": primary["coverage"],
        "selective_error": primary["selective_error"],
        "malicious_rejection_rate": primary["malicious_rejection_rate"],
        "confident_false_negative_rate": primary["confident_false_negative_rate"],
        "success": result["uncertainty"]["success_criterion"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
