"""Generate publication tables and vector/raster figures from saved research results."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from experiments.research_common import load_random_partitions
from src.evaluation.statistics import paired_effect_summary, risk_coverage_area

METRICS_DIR = Path("results/metrics")
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
ARTIFACT_DIR = Path("models/artifacts")
PALETTE = sns.color_palette("colorblind")


def load_results() -> list[dict]:
    records = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            records.append(record)
    return records


def bootstrap_ci(values, confidence: float = 0.95, repeats: int = 10000, seed: int = 2026):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def sample_std(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _method_name(record: dict) -> str:
    experiment_id = record["experiment_id"]
    if experiment_id.startswith("proposed_"):
        return "Campaign-stable compact HGB"
    if experiment_id.startswith("bef_"):
        return "Benign-envelope fusion"
    names = {
        "logistic_regression": "Logistic regression (SGD)",
        "decision_tree": "Decision tree",
        "random_forest": "Random forest",
        "hist_gradient_boosting": "HistGradientBoosting",
    }
    return names.get(record["model"].get("name"), record["model"].get("name", "Unknown"))


def _summary_rows(records: list[dict], protocol_name: str) -> pd.DataFrame:
    selected = [r for r in records if r["protocol"]["name"] == protocol_name and not r["experiment_id"].startswith("heldout_")]
    groups = {}
    for record in selected:
        groups.setdefault(_method_name(record), []).append(record)
    rows = []
    for method, group in groups.items():
        values = {
            "macro_f1": [r["metrics"]["macro_f1"] for r in group],
            "mcc": [r["metrics"]["mcc"] for r in group],
            "fpr": [r["metrics"]["false_positive_rate"] for r in group],
            "brier": [r["metrics"]["brier_score"] for r in group],
            "model_bytes": [r["efficiency"]["serialized_model_bytes"] for r in group],
            "latency_ms": [r["efficiency"]["single_record_latency_ms_median"] for r in group],
            "latency_p95_ms": [r["efficiency"]["single_record_latency_ms_p95"] for r in group],
            "throughput": [r["efficiency"]["throughput_records_per_second"] for r in group],
            "peak_memory_mb": [r["efficiency"]["peak_working_set_mb"] for r in group if r["efficiency"].get("peak_working_set_mb") is not None],
            "features": [r["feature_set"]["count"] for r in group],
        }
        lower, upper = bootstrap_ci(values["macro_f1"])
        rows.append(
            {
                "method": method,
                "n_seeds": len(group),
                "macro_f1_mean": np.mean(values["macro_f1"]),
                "macro_f1_std": sample_std(values["macro_f1"]),
                "macro_f1_ci95_low": lower,
                "macro_f1_ci95_high": upper,
                "mcc_mean": np.mean(values["mcc"]),
                "fpr_mean": np.mean(values["fpr"]),
                "brier_mean": np.mean(values["brier"]),
                "features_mean": np.mean(values["features"]),
                "model_size_kib_mean": np.mean(values["model_bytes"]) / 1024,
                "single_latency_ms_median_mean": np.mean(values["latency_ms"]),
                "single_latency_ms_p95_mean": np.mean(values["latency_p95_ms"]),
                "throughput_records_s_mean": np.mean(values["throughput"]),
                "peak_working_set_mb_mean": np.mean(values["peak_memory_mb"]) if values["peak_memory_mb"] else None,
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def make_tables(records: list[dict], metadata: dict) -> dict[str, pd.DataFrame]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    random_table = _summary_rows(records, "grouped_stratified_random")
    shift_table = _summary_rows(records, "compound_chronological_campaign_shift")

    heldout_groups = {}
    for record in records:
        if record["protocol"]["name"] != "heldout_attack_binary_detection":
            continue
        key = (record["protocol"]["heldout_attack_family"], _method_name(record))
        heldout_groups.setdefault(key, []).append(record)
    heldout_rows = []
    for (family, method), group in heldout_groups.items():
        f1 = [r["metrics"]["macro_f1"] for r in group]
        recall = [r["metrics"]["per_class"]["Malicious"]["recall"] for r in group]
        fpr = [r["metrics"]["false_positive_rate"] for r in group]
        low, high = bootstrap_ci(f1)
        heldout_rows.append(
            {
                "heldout_family": family,
                "method": method,
                "n_seeds": len(group),
                "macro_f1_mean": np.mean(f1),
                "macro_f1_std": sample_std(f1),
                "macro_f1_ci95_low": low,
                "macro_f1_ci95_high": high,
                "malicious_recall_mean": np.mean(recall),
                "malicious_recall_std": sample_std(recall),
                "fpr_mean": np.mean(fpr),
            }
        )
    heldout_table = pd.DataFrame(heldout_rows).sort_values(["heldout_family", "method"])

    proposed = [r for r in records if r["experiment_id"].startswith("proposed_random_")]
    budgets = sorted({int(key) for r in proposed for key in r["feature_set"]["validation_by_budget"]})
    budget_rows = []
    for budget in budgets:
        vals = [r["feature_set"]["validation_by_budget"][str(budget)]["macro_f1"] for r in proposed]
        budget_rows.append(
            {
                "feature_count": budget,
                "validation_macro_f1_mean": np.mean(vals),
                "validation_macro_f1_std": sample_std(vals),
            }
        )
    budget_table = pd.DataFrame(budget_rows)

    ablation_rows = []
    for name in proposed[0]["operating_points"] if proposed else []:
        group = [r["operating_points"][name] for r in proposed]
        ablation_rows.append(
            {
                "variant": name,
                "macro_f1_mean": np.mean([v["macro_f1"] for v in group]),
                "macro_f1_std": sample_std([v["macro_f1"] for v in group]),
                "fpr_mean": np.mean([v["false_positive_rate"] for v in group]),
                "brier_mean": np.mean([v["brier_score"] for v in group]),
                "ece_mean": np.mean([v["ece_15"] for v in group]),
            }
        )
    ablation_table = pd.DataFrame(ablation_rows)

    dataset_rows = []
    for split, details in metadata["split_counts"].items():
        dataset_rows.append(
            {
                "partition": split,
                "rows": details["rows"],
                "benign": details["binary_counts"].get("0", 0),
                "malicious": details["binary_counts"].get("1", 0),
                **{f"class_{key}": value for key, value in details["family_counts"].items()},
            }
        )
    dataset_table = pd.DataFrame(dataset_rows)

    rt_random = _summary_rows(records, "rt_iot2022_grouped_stratified_random")
    rt_heldout_groups = {}
    for record in records:
        if record["protocol"]["name"] != "rt_iot2022_heldout_attack_binary":
            continue
        key = (record["protocol"]["heldout_attack_family"], _method_name(record))
        rt_heldout_groups.setdefault(key, []).append(record)
    rt_heldout_rows = []
    for (family, method), group in rt_heldout_groups.items():
        f1 = [r["metrics"]["macro_f1"] for r in group]
        recall = [r["metrics"]["per_class"]["Malicious"]["recall"] for r in group]
        low, high = bootstrap_ci(f1)
        rt_heldout_rows.append(
            {
                "heldout_family": family,
                "method": method,
                "n_seeds": len(group),
                "macro_f1_mean": np.mean(f1),
                "macro_f1_std": sample_std(f1),
                "macro_f1_ci95_low": low,
                "macro_f1_ci95_high": high,
                "malicious_recall_mean": np.mean(recall),
                "malicious_recall_std": sample_std(recall),
                "fpr_mean": np.mean([r["metrics"]["false_positive_rate"] for r in group]),
            }
        )
    rt_heldout = pd.DataFrame(rt_heldout_rows).sort_values(["heldout_family", "method"])

    uncertainty_rows = []
    uncertainty_method_names = {
        "no_abstention": "No abstention",
        "confidence_target_90": "Confidence threshold",
        "global_conformal_alpha_10": "Global conformal",
        "mondrian_conformal_alpha_10": "Mondrian conformal",
    }
    for record in records:
        if not record["experiment_id"].startswith("uncertainty_"):
            continue
        for method_key, method_metrics in record["uncertainty"]["methods"].items():
            uncertainty_rows.append(
                {
                    "dataset": record["dataset"]["name"],
                    "base_protocol": record["protocol"]["base_protocol"],
                    "heldout_family": record["protocol"].get("heldout_attack_family"),
                    "model": record["model"]["name"],
                    "seed": record["seed"],
                    "method_key": method_key,
                    "method": uncertainty_method_names[method_key],
                    "coverage": method_metrics["coverage"],
                    "selective_error": method_metrics["selective_error"],
                    "selective_macro_f1": method_metrics["selective_macro_f1"],
                    "benign_rejection_rate": method_metrics["benign_rejection_rate"],
                    "malicious_rejection_rate": method_metrics["malicious_rejection_rate"],
                    "confident_false_negative_rate": method_metrics["confident_false_negative_rate"],
                    "accepted_false_positive_rate": method_metrics["accepted_false_positive_rate"],
                    "security_alert_recall": 1.0 - method_metrics["confident_false_negative_rate"],
                    "security_alert_false_positive_rate": (
                        method_metrics["benign_rejection_rate"]
                        + method_metrics["accepted_false_positive_rate"]
                    ),
                    "base_error_rejection_rate": method_metrics["base_error_rejection_rate"],
                    "set_coverage": method_metrics.get("set_coverage"),
                    "mean_set_size": method_metrics.get("mean_set_size"),
                }
            )
    uncertainty_raw = pd.DataFrame(uncertainty_rows)
    uncertainty_group_columns = [
        "dataset", "base_protocol", "heldout_family", "model", "method_key", "method"
    ]
    uncertainty_summary_rows = []
    if not uncertainty_raw.empty:
        for keys, group in uncertainty_raw.groupby(uncertainty_group_columns, dropna=False):
            row = dict(zip(uncertainty_group_columns, keys))
            row["n_seeds"] = len(group)
            for metric in (
                "coverage",
                "selective_error",
                "selective_macro_f1",
                "benign_rejection_rate",
                "malicious_rejection_rate",
                "confident_false_negative_rate",
                "accepted_false_positive_rate",
                "security_alert_recall",
                "security_alert_false_positive_rate",
                "base_error_rejection_rate",
                "set_coverage",
                "mean_set_size",
            ):
                values = group[metric].dropna().to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                row[f"{metric}_std"] = sample_std(values) if len(values) else None
            uncertainty_summary_rows.append(row)
    uncertainty_summary = pd.DataFrame(uncertainty_summary_rows)
    uncertainty_random = uncertainty_summary[
        uncertainty_summary["base_protocol"] == "random"
    ].copy() if not uncertainty_summary.empty else uncertainty_summary
    uncertainty_shift = uncertainty_summary[
        uncertainty_summary["base_protocol"] == "chronological"
    ].copy() if not uncertainty_summary.empty else uncertainty_summary
    uncertainty_heldout = uncertainty_summary[
        uncertainty_summary["base_protocol"] == "heldout"
    ].copy() if not uncertainty_summary.empty else uncertainty_summary

    density_method_names = {
        "no_abstention": "No abstention",
        "mondrian_conformal_alpha_10": "Probability Mondrian",
        "distance_mondrian_alpha_10": "Distance Mondrian",
        "fused_distance_probability_mondrian_alpha_10": "Fixed 50/50 fusion",
    }
    density_rows = []
    for record in records:
        if not record["experiment_id"].startswith("density_uncertainty_"):
            continue
        for method_key, method_name in density_method_names.items():
            metrics = record["uncertainty"]["methods"][method_key]
            density_rows.append(
                {
                    "dataset": record["dataset"]["name"],
                    "base_protocol": record["protocol"]["base_protocol"],
                    "heldout_family": record["protocol"].get("heldout_attack_family"),
                    "seed": record["seed"],
                    "method_key": method_key,
                    "method": method_name,
                    "coverage": metrics["coverage"],
                    "selective_error": metrics["selective_error"],
                    "benign_rejection_rate": metrics["benign_rejection_rate"],
                    "malicious_rejection_rate": metrics["malicious_rejection_rate"],
                    "confident_false_negative_rate": metrics["confident_false_negative_rate"],
                    "automatic_malicious_detection_rate": metrics["automatic_malicious_detection_rate"],
                    "analyst_escalations_per_10000_flows": metrics["analyst_escalations_per_10000_flows"],
                    "serialized_model_kib": record["efficiency"]["serialized_model_bytes"] / 1024,
                    "single_record_latency_ms": record["efficiency"]["single_record_latency_ms_median"],
                }
            )
    density_raw = pd.DataFrame(density_rows)
    density_summary_rows = []
    if not density_raw.empty:
        group_columns = ["dataset", "base_protocol", "heldout_family", "method_key", "method"]
        for keys, group in density_raw.groupby(group_columns, dropna=False):
            row = dict(zip(group_columns, keys))
            row["n_seeds"] = len(group)
            for metric in (
                "coverage", "selective_error", "benign_rejection_rate",
                "malicious_rejection_rate", "confident_false_negative_rate",
                "automatic_malicious_detection_rate",
                "analyst_escalations_per_10000_flows", "serialized_model_kib",
                "single_record_latency_ms",
            ):
                values = group[metric].dropna().to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                row[f"{metric}_std"] = sample_std(values) if len(values) else None
            density_summary_rows.append(row)
    density_summary = pd.DataFrame(density_summary_rows)

    paired_rows = []
    if not density_raw.empty:
        comparison_specs = [
            ("Probability Mondrian vs no abstention", "no_abstention", "mondrian_conformal_alpha_10"),
            ("Fixed fusion vs probability Mondrian", "mondrian_conformal_alpha_10", "fused_distance_probability_mondrian_alpha_10"),
        ]
        paired_metrics = (
            "confident_false_negative_rate",
            "benign_rejection_rate",
            "malicious_rejection_rate",
            "automatic_malicious_detection_rate",
        )
        cell_columns = ["dataset", "base_protocol", "heldout_family"]
        for cell_keys, cell in density_raw.groupby(cell_columns, dropna=False):
            for comparison, baseline_key, candidate_key in comparison_specs:
                baseline = cell[cell["method_key"] == baseline_key].set_index("seed")
                candidate = cell[cell["method_key"] == candidate_key].set_index("seed")
                common_seeds = sorted(set(baseline.index) & set(candidate.index))
                if len(common_seeds) < 2:
                    continue
                for metric in paired_metrics:
                    effect = paired_effect_summary(
                        baseline.loc[common_seeds, metric].to_numpy(dtype=float),
                        candidate.loc[common_seeds, metric].to_numpy(dtype=float),
                    )
                    paired_rows.append(
                        {
                            **dict(zip(cell_columns, cell_keys)),
                            "comparison": comparison,
                            "baseline_method_key": baseline_key,
                            "candidate_method_key": candidate_key,
                            "metric": metric,
                            "difference_definition": "candidate_minus_baseline",
                            **effect,
                        }
                    )
    paired_table = pd.DataFrame(paired_rows)

    aurc_rows = []
    curve_specs = (
        ("uncertainty_", "confidence", "Confidence threshold"),
        ("uncertainty_", "mondrian_conformal", "Probability Mondrian"),
        ("density_uncertainty_", "fused_mondrian", "Fixed 50/50 fusion"),
    )
    for record in records:
        for prefix, curve_key, method_name in curve_specs:
            if not record["experiment_id"].startswith(prefix):
                continue
            if prefix == "uncertainty_" and record["model"]["name"] != "hgb":
                continue
            curve = record["uncertainty"]["curves"].get(curve_key, [])
            points = [point for point in curve if point.get("selective_error") is not None]
            if len(points) < 2:
                continue
            area = risk_coverage_area(
                np.asarray([point["coverage"] for point in points]),
                np.asarray([point["selective_error"] for point in points]),
            )
            aurc_rows.append(
                {
                    "dataset": record["dataset"]["name"],
                    "base_protocol": record["protocol"]["base_protocol"],
                    "heldout_family": record["protocol"].get("heldout_attack_family"),
                    "seed": record["seed"],
                    "method": method_name,
                    **area,
                }
            )
    aurc_raw = pd.DataFrame(aurc_rows)
    aurc_summary_rows = []
    if not aurc_raw.empty:
        group_columns = ["dataset", "base_protocol", "heldout_family", "method"]
        for keys, group in aurc_raw.groupby(group_columns, dropna=False):
            row = dict(zip(group_columns, keys))
            row["n_seeds"] = len(group)
            for metric in (
                "coverage_min", "coverage_max", "coverage_span",
                "risk_coverage_area", "mean_risk_over_coverage_span",
            ):
                values = group[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = sample_std(values)
            aurc_summary_rows.append(row)
    aurc_summary = pd.DataFrame(aurc_summary_rows)

    cross_rows, cross_family_rows = [], []
    for record in records:
        if not record["experiment_id"].startswith("cross_netflow_"):
            continue
        metrics = record["metrics"]
        source_metrics = record["source_validation_metrics"]
        mondrian = record["uncertainty"]["methods"]["mondrian_conformal_alpha_10"]
        cross_rows.append(
            {
                "source": record["protocol"]["source"],
                "target": record["protocol"]["target"],
                "model": record["model"]["name"],
                "seed": record["seed"],
                "source_validation_macro_f1": source_metrics["macro_f1"],
                "source_validation_malicious_recall": source_metrics["per_class"]["Malicious"]["recall"],
                "target_macro_f1": metrics["macro_f1"],
                "target_malicious_recall": metrics["per_class"]["Malicious"]["recall"],
                "target_fpr": metrics["false_positive_rate"],
                "target_mcc": metrics["mcc"],
                "mondrian_coverage": mondrian["coverage"],
                "mondrian_confident_false_negative_rate": mondrian["confident_false_negative_rate"],
                "mondrian_benign_rejection_rate": mondrian["benign_rejection_rate"],
                "mondrian_malicious_rejection_rate": mondrian["malicious_rejection_rate"],
                "mondrian_automatic_malicious_detection_rate": mondrian["automatic_malicious_detection_rate"],
                "serialized_model_kib": record["efficiency"]["serialized_model_bytes"] / 1024,
                "single_record_latency_ms": record["efficiency"]["single_record_latency_ms_median"],
                "cross_overlap_rows_removed": record["protocol"]["cross_dataset_overlap_rows_removed_from_target"],
            }
        )
        for family, family_metrics in record["target_family_uncertainty"].items():
            cross_family_rows.append(
                {
                    "source": record["protocol"]["source"],
                    "target": record["protocol"]["target"],
                    "model": record["model"]["name"],
                    "seed": record["seed"],
                    "family": family,
                    **family_metrics,
                }
            )
    cross_raw = pd.DataFrame(cross_rows)
    cross_summary_rows = []
    if not cross_raw.empty:
        group_columns = ["source", "target", "model"]
        for keys, group in cross_raw.groupby(group_columns):
            row = dict(zip(group_columns, keys))
            row["n_seeds"] = len(group)
            for metric in [column for column in cross_raw.columns if column not in group_columns + ["seed"]]:
                values = group[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = sample_std(values)
            cross_summary_rows.append(row)
    cross_summary = pd.DataFrame(cross_summary_rows)
    cross_family_raw = pd.DataFrame(cross_family_rows)
    cross_family_summary_rows = []
    if not cross_family_raw.empty:
        group_columns = ["source", "target", "model", "family"]
        for keys, group in cross_family_raw.groupby(group_columns):
            row = dict(zip(group_columns, keys))
            row["n_seeds"] = len(group)
            for metric in [column for column in cross_family_raw.columns if column not in group_columns + ["seed"]]:
                values = group[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = sample_std(values)
            cross_family_summary_rows.append(row)
    cross_family_summary = pd.DataFrame(cross_family_summary_rows)

    tables = {
        "research_random_baselines.csv": random_table,
        "research_shift_results.csv": shift_table,
        "research_heldout_attacks.csv": heldout_table,
        "research_feature_budgets.csv": budget_table,
        "research_ablation.csv": ablation_table,
        "research_dataset_characteristics.csv": dataset_table,
        "research_rt_iot2022_random.csv": rt_random,
        "research_rt_iot2022_heldout.csv": rt_heldout,
        "research_uncertainty_random.csv": uncertainty_random,
        "research_uncertainty_shift.csv": uncertainty_shift,
        "research_uncertainty_heldout.csv": uncertainty_heldout,
        "research_density_ablation.csv": density_summary,
        "research_paired_effects.csv": paired_table,
        "research_risk_coverage_summary.csv": aurc_summary,
        "research_cross_netflow_summary.csv": cross_summary,
        "research_cross_netflow_family.csv": cross_family_summary,
    }
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / name, index=False)
    return tables


def _save_figure(figure, stem: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(FIGURES_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def class_distribution_figure(metadata: dict):
    counts = metadata["split_counts"]["train"]["family_counts"]
    frame = pd.DataFrame({"Class": list(counts), "Count": list(counts.values())}).sort_values("Count")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(frame["Class"], frame["Count"], color=PALETTE[0])
    ax.set_xscale("log")
    ax.set_xlabel("Training records (log scale)")
    ax.set_ylabel("")
    ax.set_title("Corrected CICIDS2017 remains strongly imbalanced after cleaning")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, "research_figure1_class_distribution")


def pareto_figure(random_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for index, row in random_table.reset_index(drop=True).iterrows():
        ax.scatter(row["model_size_kib_mean"], row["macro_f1_mean"], s=70, color=PALETTE[index % len(PALETTE)])
        right_side = row["model_size_kib_mean"] > 1000
        ax.annotate(
            row["method"],
            (row["model_size_kib_mean"], row["macro_f1_mean"]),
            xytext=(-5 if right_side else 5, 5),
            textcoords="offset points",
            fontsize=8,
            ha="right" if right_side else "left",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Serialized model size (KiB, log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Random-split performance–size trade-off (five seeds)")
    ax.grid(alpha=0.25)
    _save_figure(fig, "research_figure2_performance_size_pareto")


def feature_budget_figure(table: pd.DataFrame):
    table = table.sort_values("feature_count")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(
        table["feature_count"],
        table["validation_macro_f1_mean"],
        yerr=table["validation_macro_f1_std"],
        marker="o",
        capsize=4,
        color=PALETTE[0],
    )
    ax.set_xlabel("Selected feature count")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Campaign-stable feature-budget selection")
    ax.grid(alpha=0.25)
    _save_figure(fig, "research_figure3_feature_budget")


def shift_figure(random_table: pd.DataFrame, shift_table: pd.DataFrame):
    common = sorted(set(random_table["method"]) & set(shift_table["method"]))
    rows = []
    for method in common:
        rows.append({"Method": method, "Protocol": "Grouped random", "Macro-F1": float(random_table.loc[random_table.method == method, "macro_f1_mean"].iloc[0])})
        rows.append({"Method": method, "Protocol": "Mon-Wed → Friday", "Macro-F1": float(shift_table.loc[shift_table.method == method, "macro_f1_mean"].iloc[0])})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.barplot(data=frame, y="Method", x="Macro-F1", hue="Protocol", palette=PALETTE[:2], ax=ax)
    ax.set_xlim(0.70, 1.01)
    ax.set_ylabel("")
    ax.set_xlabel("Test macro-F1")
    ax.set_title("Random splits substantially overstate unseen-campaign performance")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, "research_figure4_random_vs_shift")


def heldout_figure(table: pd.DataFrame):
    families = table["heldout_family"].drop_duplicates().tolist()
    methods = table["method"].drop_duplicates().tolist()
    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for index, method in enumerate(methods):
        subset = table[table["method"] == method].set_index("heldout_family").reindex(families)
        ax.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            subset["malicious_recall_mean"],
            width,
            yerr=subset["malicious_recall_std"],
            capsize=3,
            label=method,
            color=PALETTE[index],
        )
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, families)
    ax.set_xlabel("Attack family absent from development data")
    ax.set_ylabel("Malicious recall")
    ax.set_title("Transfer to unseen attack families is family-dependent")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Method", fontsize=8)
    _save_figure(fig, "research_figure5_heldout_attack_recall")


def rt_iot2022_figure(random_table: pd.DataFrame, heldout_table: pd.DataFrame):
    method_order = random_table["method"].tolist()
    families = heldout_table["heldout_family"].drop_duplicates().tolist()
    labels = ["Random test"] + families
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for index, method in enumerate(method_order):
        random_f1 = float(random_table.loc[random_table.method == method, "macro_f1_mean"].iloc[0])
        subset = heldout_table[heldout_table.method == method].set_index("heldout_family").reindex(families)
        values = [random_f1] + subset["macro_f1_mean"].tolist()
        errors = [float(random_table.loc[random_table.method == method, "macro_f1_std"].iloc[0])] + subset["macro_f1_std"].tolist()
        ax.bar(
            x + (index - (len(method_order) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=method,
            color=PALETTE[index],
        )
    ax.set_ylim(0.45, 1.01)
    display_names = {
        "DOS_SYN_Hping": "SYN flood",
        "ARP_poisioning": "ARP poisoning",
        "NMAP_UDP_SCAN": "Nmap UDP",
        "DDOS_Slowloris": "Slowloris",
    }
    ax.set_xticks(x, ["Random test"] + [display_names.get(name, name) for name in families], rotation=10)
    ax.set_ylabel("Binary macro-F1")
    ax.set_xlabel("RT-IoT2022 evaluation setting")
    ax.set_title("Independent IoT replication confirms unseen-attack degradation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(fig, "research_figure11_rt_iot2022_replication")


def uncertainty_risk_coverage_figure(records: list[dict]):
    settings = [
        ("corrected_cicids2017", "random", "CICIDS grouped random"),
        ("corrected_cicids2017", "chronological", "CICIDS campaign shift"),
        ("RT-IoT2022", "random", "RT-IoT2022 grouped random"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=False)
    for ax, (dataset, protocol, title) in zip(axes, settings):
        selected = [
            record for record in records
            if record["experiment_id"].startswith("uncertainty_")
            and record["dataset"]["name"] == dataset
            and record["protocol"]["base_protocol"] == protocol
            and record["model"]["name"] == "hgb"
        ]
        for method_index, (curve_key, label) in enumerate(
            (("confidence", "Confidence"), ("mondrian_conformal", "Mondrian conformal"))
        ):
            by_position = {}
            for record in selected:
                for position, point in enumerate(record["uncertainty"]["curves"][curve_key]):
                    by_position.setdefault(position, []).append(point)
            points = []
            for position in sorted(by_position):
                group = by_position[position]
                errors = [point["selective_error"] for point in group if point["selective_error"] is not None]
                if errors:
                    points.append(
                        (
                            np.mean([point["coverage"] for point in group]),
                            np.mean(errors),
                        )
                    )
            if points:
                points = sorted(points)
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    markersize=3,
                    label=label,
                    color=PALETTE[method_index],
                )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Accepted coverage")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Selective error among accepted flows")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Risk--coverage behavior of validation-calibrated rejection", y=1.02)
    _save_figure(fig, "research_figure12_uncertainty_risk_coverage")


def uncertainty_heldout_figure(table: pd.DataFrame):
    methods = ["No abstention", "Confidence threshold", "Mondrian conformal"]
    datasets = ["corrected_cicids2017", "RT-IoT2022"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, dataset in zip(axes, datasets):
        subset = table[(table["dataset"] == dataset) & table["method"].isin(methods)]
        families = subset["heldout_family"].drop_duplicates().tolist()
        x = np.arange(len(families))
        width = 0.25
        for method_index, method in enumerate(methods):
            values = (
                subset[subset["method"] == method]
                .set_index("heldout_family")
                .reindex(families)["confident_false_negative_rate_mean"]
            )
            ax.bar(
                x + (method_index - 1) * width,
                values,
                width,
                label=method,
                color=PALETTE[method_index],
            )
        ax.set_xticks(x, families, rotation=20, ha="right", fontsize=7)
        ax.set_title("Corrected CICIDS2017" if dataset.startswith("corrected") else "RT-IoT2022")
        ax.set_xlabel("Attack absent from development")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Confident false negatives / malicious flows")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Abstention can convert some unseen-attack misses into unknown alerts", y=1.02)
    _save_figure(fig, "research_figure13_uncertainty_heldout_confident_fn")


def density_ablation_figure(table: pd.DataFrame):
    methods = ["Probability Mondrian", "Fixed 50/50 fusion"]
    datasets = ["corrected_cicids2017", "RT-IoT2022"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharey="row")
    display_names = {
        "ARP_poisioning": "ARP poisoning",
        "DDOS_Slowloris": "Slowloris",
        "DOS_SYN_Hping": "SYN Hping",
        "NMAP_UDP_SCAN": "Nmap UDP",
    }
    for column, dataset in enumerate(datasets):
        subset = table[
            (table["dataset"] == dataset)
            & (table["base_protocol"] == "heldout")
            & table["method"].isin(methods)
        ]
        families = subset["heldout_family"].drop_duplicates().tolist()
        x = np.arange(len(families))
        width = 0.36
        for method_index, method in enumerate(methods):
            indexed = subset[subset["method"] == method].set_index("heldout_family").reindex(families)
            offset = (method_index - 0.5) * width
            axes[0, column].bar(
                x + offset,
                indexed["confident_false_negative_rate_mean"],
                width,
                label=method,
                color=PALETTE[method_index],
            )
            axes[1, column].bar(
                x + offset,
                indexed["malicious_rejection_rate_mean"],
                width,
                label=method,
                color=PALETTE[method_index],
            )
        title = "Corrected CICIDS2017" if dataset.startswith("corrected") else "RT-IoT2022"
        axes[0, column].set_title(title)
        for row in range(2):
            axes[row, column].set_xticks(
                x,
                [display_names.get(name, name) for name in families],
                rotation=20,
                ha="right",
                fontsize=7,
            )
            axes[row, column].grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Confident FN / malicious flows")
    axes[1, 0].set_ylabel("Rejected / malicious flows")
    axes[1, 0].set_xlabel("Attack absent from development")
    axes[1, 1].set_xlabel("Attack absent from development")
    axes[0, 1].legend(fontsize=7)
    fig.suptitle("Naive distance fusion trades rejection for confident unseen-attack misses", y=1.01)
    _save_figure(fig, "research_figure14_density_ablation")


def density_tradeoff_figure(table: pd.DataFrame):
    subset = table[
        (table["base_protocol"] == "heldout")
        & table["method"].isin(["Probability Mondrian", "Fixed 50/50 fusion"])
    ]
    probability = subset[subset["method"] == "Probability Mondrian"].set_index(
        ["dataset", "heldout_family"]
    )
    fusion = subset[subset["method"] == "Fixed 50/50 fusion"].set_index(
        ["dataset", "heldout_family"]
    )
    common = probability.index.intersection(fusion.index)
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    markers = {"corrected_cicids2017": "o", "RT-IoT2022": "s"}
    display_names = {
        "ARP_poisioning": "ARP",
        "DDOS_Slowloris": "Slowloris",
        "DOS_SYN_Hping": "SYN Hping",
        "NMAP_UDP_SCAN": "Nmap UDP",
        "Brute Force": "Brute Force",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "Web Attack": "Web Attack",
    }
    for dataset_index, dataset in enumerate(markers):
        points = [index for index in common if index[0] == dataset]
        x = [
            fusion.loc[index, "malicious_rejection_rate_mean"]
            - probability.loc[index, "malicious_rejection_rate_mean"]
            for index in points
        ]
        y = [
            fusion.loc[index, "confident_false_negative_rate_mean"]
            - probability.loc[index, "confident_false_negative_rate_mean"]
            for index in points
        ]
        ax.scatter(
            x, y, s=65, marker=markers[dataset], color=PALETTE[dataset_index],
            label="Corrected CICIDS2017" if dataset.startswith("corrected") else "RT-IoT2022",
        )
        for point, x_value, y_value in zip(points, x, y):
            ax.annotate(display_names.get(point[1], point[1]), (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.margins(x=0.12, y=0.12)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.axvline(0, color="0.35", linewidth=1)
    x_min, _ = ax.get_xlim()
    y_min, _ = ax.get_ylim()
    ax.fill_between([x_min, 0], y_min, 0, color=PALETTE[2], alpha=0.08, zorder=0)
    ax.set_xlabel("Δ malicious rejection rate")
    ax.set_ylabel("Δ confident false-negative rate")
    ax.set_title("Fusion − probability Mondrian: held-out trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(fig, "research_figure15_density_tradeoff")


def cross_netflow_transfer_figure(table: pd.DataFrame):
    display = {
        "nf_unsw_nb15_v2": "NF-UNSW-NB15-v2",
        "nf_ton_iot_v2": "NF-ToN-IoT-v2",
        "decision_tree": "Decision tree",
        "hgb": "HGB",
    }
    labels = [
        f"{display[row.source]} →\n{display[row.target]}\n{display[row.model]}"
        for row in table.itertuples()
    ]
    x = np.arange(len(table))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    metric_pairs = [
        ("source_validation_macro_f1_mean", "target_macro_f1_mean", "Macro-F1"),
        ("source_validation_malicious_recall_mean", "target_malicious_recall_mean", "Malicious recall"),
    ]
    for ax, (source_metric, target_metric, title) in zip(axes, metric_pairs):
        ax.bar(x - width / 2, table[source_metric], width, label="Source validation", color=PALETTE[0])
        ax.bar(x + width / 2, table[target_metric], width, label="Cross-dataset target", color=PALETTE[1])
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(fontsize=8)
    fig.suptitle("Shared NetFlow features do not ensure cross-testbed transfer", y=1.02)
    _save_figure(fig, "research_figure16_cross_netflow_transfer")


def cross_netflow_family_figure(table: pd.DataFrame):
    table = table[table["model"] == "hgb"].copy()
    directions = list(table[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), sharex=True)
    display_direction = {
        ("nf_unsw_nb15_v2", "nf_ton_iot_v2"): "NF-UNSW → NF-ToN-IoT",
        ("nf_ton_iot_v2", "nf_unsw_nb15_v2"): "NF-ToN-IoT → NF-UNSW",
    }
    for ax, direction in zip(axes, directions):
        subset = table[(table["source"] == direction[0]) & (table["target"] == direction[1])].sort_values("family")
        y = np.arange(len(subset))
        automatic = subset["mondrian_automatic_malicious_detection_rate_mean"].to_numpy()
        rejected = subset["mondrian_malicious_rejection_rate_mean"].to_numpy()
        missed = subset["mondrian_confident_false_negative_rate_mean"].to_numpy()
        ax.barh(y, automatic, label="Automatic malicious", color=PALETTE[2])
        ax.barh(y, rejected, left=automatic, label="Unknown/rejected", color=PALETTE[1])
        ax.barh(y, missed, left=automatic + rejected, label="Confident miss", color=PALETTE[3])
        ax.set_yticks(y, subset["family"], fontsize=7)
        ax.set_title(display_direction.get(direction, f"{direction[0]} → {direction[1]}"))
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_xlabel("Fraction of target-family malicious flows")
    axes[1].set_xlabel("Fraction of target-family malicious flows")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Source-calibrated HGB abstention remains confidently wrong across testbeds", y=1.02)
    _save_figure(fig, "research_figure17_cross_netflow_family")


def representative_probability_figures(metadata: dict):
    feature_names = metadata["feature_columns"]
    partitions = load_random_partitions(
        Path("data/processed/research_cicids2017.parquet"),
        feature_names,
        {"train": 300000, "validation": 75000, "test": 100000},
        seed=0,
    )
    test = partitions["test"]
    X_test = test[feature_names].to_numpy(dtype=np.float32)
    y_test = test["binary_label"].to_numpy(dtype=np.int8)
    models = {
        "Full HGB": joblib.load(ARTIFACT_DIR / "baseline_random_hist_gradient_boosting_seed0.joblib"),
        "Compact calibrated HGB": joblib.load(ARTIFACT_DIR / "proposed_random_campaign_stable_hist_gradient_boosting_seed0.joblib"),
    }
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    for index, (name, model) in enumerate(models.items()):
        probability = model.predict_proba(X_test)[:, 1]
        observed, predicted = calibration_curve(y_test, probability, n_bins=12, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name, color=PALETTE[index])
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="Perfect calibration")
    ax.set_xlabel("Mean predicted malicious probability")
    ax.set_ylabel("Observed malicious frequency")
    ax.set_title("Reliability diagram (representative seed 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save_figure(fig, "research_figure6_reliability")

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    prevalence = y_test.mean()
    for index, (name, model) in enumerate(models.items()):
        probability = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ax.plot(recall, precision, label=name, color=PALETTE[index])
    ax.axhline(prevalence, linestyle="--", color="0.4", label=f"Prevalence ({prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves (representative seed 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save_figure(fig, "research_figure7_precision_recall")


def confusion_figure(records: list[dict]):
    record = next(r for r in records if r["experiment_id"] == "proposed_random_campaign_stable_hist_gradient_boosting_seed0")
    matrix = np.asarray(record["metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Compact pipeline confusion matrix (seed 0)")
    _save_figure(fig, "research_figure8_confusion_matrix")


def feature_importance_figure(records: list[dict]):
    proposed = [r for r in records if r["experiment_id"].startswith("proposed_random_")]
    scores = {}
    for record in proposed:
        for item in record["feature_set"]["ranking"]:
            scores.setdefault(item["feature"], []).append(item["stability_score"])
    frame = pd.DataFrame(
        [{"Feature": key, "Score": np.mean(value)} for key, value in scores.items()]
    ).nlargest(15, "Score").sort_values("Score")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(frame["Feature"], frame["Score"], color=PALETTE[2])
    ax.set_xlabel("Mean campaign-stability score")
    ax.set_ylabel("")
    ax.set_title("Top training-only campaign-stable features")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, "research_figure9_feature_importance")


def methodology_figure():
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    labels = [
        "Corrected CICIDS2017\n+ RT-IoT2022",
        "Contradiction removal\n+ provenance",
        "Frozen grouped /\nday / held-out splits",
        "Lightweight models\n+ stable features",
        "Calibration +\nvalidation threshold",
        "Accuracy, reliability,\nshift and efficiency",
    ]
    x_positions = np.linspace(0.03, 0.84, len(labels))
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        box = FancyBboxPatch((x, 0.36), 0.13, 0.30, boxstyle="round,pad=0.02", facecolor=PALETTE[index % len(PALETTE)], alpha=0.25, edgecolor=PALETTE[index % len(PALETTE)], linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.065, 0.51, label, ha="center", va="center", fontsize=8)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x_positions[index + 1] - 0.008, 0.51), xytext=(x + 0.138, 0.51), arrowprops={"arrowstyle": "->", "color": "0.3"})
    ax.text(0.5, 0.88, "Leakage-aware, resource-controlled research workflow", ha="center", fontsize=12, weight="bold")
    _save_figure(fig, "research_figure10_methodology")


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    records = load_results()
    metadata = json.loads(Path("data/processed/research_metadata.json").read_text(encoding="utf-8"))
    tables = make_tables(records, metadata)
    class_distribution_figure(metadata)
    pareto_figure(tables["research_random_baselines.csv"])
    feature_budget_figure(tables["research_feature_budgets.csv"])
    shift_figure(tables["research_random_baselines.csv"], tables["research_shift_results.csv"])
    heldout_figure(tables["research_heldout_attacks.csv"])
    rt_iot2022_figure(
        tables["research_rt_iot2022_random.csv"],
        tables["research_rt_iot2022_heldout.csv"],
    )
    if any(record["experiment_id"].startswith("uncertainty_") for record in records):
        uncertainty_risk_coverage_figure(records)
        uncertainty_heldout_figure(tables["research_uncertainty_heldout.csv"])
    if any(record["experiment_id"].startswith("density_uncertainty_") for record in records):
        density_ablation_figure(tables["research_density_ablation.csv"])
        density_tradeoff_figure(tables["research_density_ablation.csv"])
    if any(record["experiment_id"].startswith("cross_netflow_") for record in records):
        cross_netflow_transfer_figure(tables["research_cross_netflow_summary.csv"])
        cross_netflow_family_figure(tables["research_cross_netflow_family.csv"])
    representative_probability_figures(metadata)
    confusion_figure(records)
    feature_importance_figure(records)
    methodology_figure()
    figure_count = 13
    if any(record["experiment_id"].startswith("density_uncertainty_") for record in records):
        figure_count += 2
    if any(record["experiment_id"].startswith("cross_netflow_") for record in records):
        figure_count += 2
    print(f"Generated {len(tables)} tables and {figure_count} figures from {len(records)} result records.")


if __name__ == "__main__":
    main()
