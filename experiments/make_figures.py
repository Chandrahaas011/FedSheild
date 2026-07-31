"""Regenerates results/figures/*.png from results/runs/*.json and results/shap/*.json.

    python -m experiments.make_figures

Palette: Okabe-Ito colorblind-safe categorical subset, validated with the dataviz
skill's validator (all six checks pass for light mode) — fixed hue order, never cycled:
Centralized=blue, FedAvg=orange, Your-method=green, Your-method+DP=vermillion.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("results/runs")
SHAP_DIR = Path("results/shap")
OUT_DIR = Path("results/figures")
SEEDS = [0, 1, 2]

METHOD_COLORS = {
    "centralized": "#0072B2",
    "fedavg": "#E69F00",
    "topk": "#009E73",
    "topk_dp": "#D55E00",
}
METHOD_LABELS = {
    "centralized": "Centralized",
    "fedavg": "FedAvg",
    "topk": "Your-method (top-k)",
    "topk_dp": "Your-method+DP (flagship)",
}
DIVERGING_POS = "#D55E00"  # pushes toward attack prediction
DIVERGING_NEG = "#0072B2"  # pushes toward benign prediction


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.tick_params(colors="#555555")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def load_run(run_id: str) -> dict | None:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _mean_std_over_seeds(method: str, key: str):
    """Aligns per-seed histories on round number and returns (rounds, mean, std)."""
    per_round = {}
    for seed in SEEDS:
        run = load_run(f"{method}_alpha0.5_seed{seed}")
        if run is None:
            continue
        h = run["history"]
        for r, v in zip(h["rounds"], h[key]):
            if v is not None:
                per_round.setdefault(r, []).append(v)
    rounds = sorted(per_round.keys())
    means = np.array([np.mean(per_round[r]) for r in rounds])
    stds = np.array([np.std(per_round[r]) for r in rounds])
    return np.array(rounds), means, stds


def figure1_convergence():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ["fedavg", "topk", "topk_dp"]:
        rounds, means, stds = _mean_std_over_seeds(method, "server_test_macro_f1")
        if len(rounds) == 0:
            continue
        color = METHOD_COLORS[method]
        ax.plot(rounds, means, color=color, linewidth=2, label=METHOD_LABELS[method], zorder=3)
        ax.fill_between(rounds, means - stds, means + stds, color=color, alpha=0.15, linewidth=0, zorder=2)

    cent_f1s = [load_run(f"centralized_seed{s}")["final_metrics"]["macro_f1"]
                for s in SEEDS if load_run(f"centralized_seed{s}")]
    if cent_f1s:
        ax.axhline(np.mean(cent_f1s), color=METHOD_COLORS["centralized"], linestyle="--", linewidth=1.5,
                   label=f"{METHOD_LABELS['centralized']} (upper bound)", zorder=3)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Macro-F1 (held-out test set)")
    ax.set_title("Convergence: macro-F1 vs. communication rounds")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure1_convergence.png", dpi=150)
    plt.close(fig)


def figure2_communication_savings():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ["fedavg", "topk"]:
        rounds, means, stds = _mean_std_over_seeds(method, "cumulative_bytes")
        if len(rounds) == 0:
            continue
        means_mb, stds_mb = means / 1e6, stds / 1e6
        color = METHOD_COLORS[method]
        ax.plot(rounds, means_mb, color=color, linewidth=2, label=METHOD_LABELS[method], zorder=3)
        ax.fill_between(rounds, means_mb - stds_mb, means_mb + stds_mb, color=color, alpha=0.15, linewidth=0, zorder=2)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative bytes transmitted (MB)")
    ax.set_title("Communication cost: your-method vs. FedAvg")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2_communication_savings.png", dpi=150)
    plt.close(fig)


def figure3_shap_global_summary(shap_run_id: str, top_n: int = 10):
    path = SHAP_DIR / f"{shap_run_id}.json"
    if not path.exists():
        print(f"skip figure3: {path} not found (run experiments.run_xai first)")
        return
    with open(path) as f:
        d = json.load(f)
    feature_names = d["feature_names"]
    classes = list(d["global_summary"].keys())
    if not classes:
        print("skip figure3: no attack classes explained")
        return

    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), squeeze=False)
    for ax, cls in zip(axes[0], classes):
        vals = np.array(d["global_summary"][cls]["mean_abs_shap"])
        order = np.argsort(vals)[::-1][:top_n]
        top_feats = [feature_names[i] for i in order][::-1]
        top_vals = vals[order][::-1]
        # Sequential single-hue magnitude encoding, light->dark by rank.
        shades = plt.cm.Blues(np.linspace(0.35, 0.9, len(top_vals)))
        ax.barh(range(len(top_vals)), top_vals, color=shades, zorder=3)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(top_feats, fontsize=8)
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"{cls} (n={d['global_summary'][cls]['n_samples']})", fontsize=10)
        _style_axes(ax)
    fig.suptitle("Federated SHAP: global feature importance by attack class (server-aggregated)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure3_shap_global_summary.png", dpi=150)
    plt.close(fig)


def figure4_local_explanations(shap_run_id: str, n_examples: int = 3, top_n: int = 8):
    path = SHAP_DIR / f"{shap_run_id}.json"
    if not path.exists():
        print(f"skip figure4: {path} not found")
        return
    with open(path) as f:
        d = json.load(f)
    # Pick one example per distinct attack class (up to n_examples) rather than the
    # first N in list order, which tends to be dominated by whichever class the
    # client-iteration order happened to hit first.
    examples, seen_classes = [], set()
    for ex in d["local_examples"]:
        if ex["class"] not in seen_classes:
            examples.append(ex)
            seen_classes.add(ex["class"])
        if len(examples) == n_examples:
            break
    feature_names = d["feature_names"]
    if not examples:
        print("skip figure4: no local examples cached")
        return

    fig, axes = plt.subplots(1, len(examples), figsize=(4.5 * len(examples), 4.5), squeeze=False)
    for ax, ex in zip(axes[0], examples):
        shap_vals = np.array(ex["shap_values"])
        order = np.argsort(np.abs(shap_vals))[::-1][:top_n]
        feats = [feature_names[i] for i in order][::-1]
        vals = shap_vals[order][::-1]
        colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in vals]
        ax.barh(range(len(vals)), vals, color=colors, zorder=3)
        ax.axvline(0, color="#888888", linewidth=1, zorder=2)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Alert: {ex['class']}", fontsize=10)
        _style_axes(ax)
    fig.suptitle("Per-alert local explanations (client-side, individual flagged flows)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure4_local_explanations.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure1_convergence()
    figure2_communication_savings()
    # Non-DP top-k model: captures more attack classes than the DP-noised flagship, the
    # stronger demonstration of the federated-XAI architecture itself (ROADMAP.md Phase 4
    # gate). The flagship's own SHAP results are in results/shap/topk_dp_*.json.
    figure3_shap_global_summary("topk_alpha0.5_seed0")
    figure4_local_explanations("topk_alpha0.5_seed0")
    print("Figures written to results/figures/")


if __name__ == "__main__":
    main()
