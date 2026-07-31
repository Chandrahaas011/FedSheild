"""Federated XAI: client-side SHAP + server-side aggregation of summaries only.

This is the paper's primary novelty (ROADMAP.md Phase 4): each client runs SHAP
*locally* against the converged global model on its own flagged (predicted-attack)
traffic. Raw data and raw per-sample SHAP values never leave the client — only
mean(|SHAP|) per feature per attack class is aggregated server-side. A handful of raw
per-alert explanations are kept *client-side* purely for illustrative local-explanation
figures (Figure 4), never sent to the "server" aggregate.
"""
import numpy as np
import shap
import torch


def explain_client(
    model: torch.nn.Module, X_client: np.ndarray, feature_names: list[str], class_names: list[str],
    benign_idx: int = 0, background_size: int = 50, max_explain_per_class: int = 100,
    max_local_examples_per_class: int = 3, seed: int = 42,
) -> dict:
    """Runs SHAP on one client's local data against the (already-trained, converged)
    global model. Returns a dict with a per-class mean(|SHAP|) summary (safe to send to
    the server) and a small number of raw per-alert examples (kept client-side)."""
    rng = np.random.default_rng(seed)
    model.eval()

    if len(X_client) < 2:
        return {"summary": {}, "local_examples": []}

    bg_idx = rng.choice(len(X_client), size=min(background_size, len(X_client)), replace=False)
    background = torch.from_numpy(X_client[bg_idx])

    with torch.no_grad():
        preds = model(torch.from_numpy(X_client)).argmax(dim=1).numpy()

    flagged_idx = np.where(preds != benign_idx)[0]
    if len(flagged_idx) == 0:
        return {"summary": {}, "local_examples": []}

    explain_idx = rng.choice(flagged_idx, size=min(max_explain_per_class * 4, len(flagged_idx)), replace=False)
    X_explain = torch.from_numpy(X_client[explain_idx])

    # Note: for DP-trained models (BatchNorm replaced by GroupNorm, see src/fl/dp.py),
    # SHAP's DeepExplainer has no dedicated propagation rule for GroupNorm and falls
    # back to a generic gradient-based rule (a known SHAP limitation, logged as a
    # UserWarning) — attributions on DP runs are a reasonable approximation, not as
    # rigorously validated as the non-DP path. Worth flagging explicitly in the paper.
    explainer = shap.DeepExplainer(model, background)
    raw = explainer.shap_values(X_explain, check_additivity=False)
    shap_by_class = _normalize_shap_output(raw, len(explain_idx), len(feature_names), len(class_names))

    pred_for_explain = preds[explain_idx]
    summary = {}
    local_examples = []
    for c in np.unique(pred_for_explain):
        if c == benign_idx:
            continue
        mask = pred_for_explain == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        vals = shap_by_class[mask, :, c]
        summary[class_names[c]] = {
            "mean_abs_shap": np.abs(vals).mean(axis=0).tolist(),
            "n_samples": n_c,
        }
        example_positions = np.where(mask)[0][:max_local_examples_per_class]
        for pos in example_positions:
            local_examples.append({
                "class": class_names[c],
                "feature_values": X_explain[pos].numpy().tolist(),
                "shap_values": shap_by_class[pos, :, c].tolist(),
            })

    return {"summary": summary, "local_examples": local_examples}


def _normalize_shap_output(raw, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    """Different SHAP versions return either a list of (n_samples, n_features) arrays
    (one per class) or a single (n_samples, n_features, n_classes) array. Normalize to
    the latter."""
    if isinstance(raw, list):
        return np.stack(raw, axis=-1)
    arr = np.asarray(raw)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def aggregate_client_summaries(client_summaries: list[dict], feature_names: list[str], class_names: list[str]) -> dict:
    """Server-side: weighted-mean aggregation of client summary dicts only (no raw
    per-sample SHAP values ever seen here)."""
    n_features = len(feature_names)
    weighted_sum = {c: np.zeros(n_features) for c in class_names}
    weight_total = {c: 0 for c in class_names}

    for client_summary in client_summaries:
        for cls, d in client_summary.get("summary", {}).items():
            weighted_sum[cls] += np.array(d["mean_abs_shap"]) * d["n_samples"]
            weight_total[cls] += d["n_samples"]

    global_summary = {}
    for cls in class_names:
        if weight_total[cls] > 0:
            global_summary[cls] = {
                "mean_abs_shap": (weighted_sum[cls] / weight_total[cls]).tolist(),
                "n_samples": weight_total[cls],
            }
    return global_summary


def run_federated_shap(
    model: torch.nn.Module, X: np.ndarray, client_indices: list[np.ndarray],
    feature_names: list[str], class_names: list[str], benign_idx: int = 0, seed: int = 42,
) -> tuple[dict, list[dict]]:
    client_summaries = []
    all_local_examples = []
    for client_id, idx in enumerate(client_indices):
        if len(idx) < 2:
            continue
        result = explain_client(
            model, X[idx], feature_names, class_names, benign_idx=benign_idx, seed=seed + client_id
        )
        client_summaries.append(result)
        all_local_examples.extend(result["local_examples"])

    global_summary = aggregate_client_summaries(client_summaries, feature_names, class_names)
    return global_summary, all_local_examples
