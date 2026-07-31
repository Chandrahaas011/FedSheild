"""Feature/attack alignment scoring (ROADMAP.md Phase 4 gate): do SHAP's top features
for a given attack class match known attack signatures (e.g. flow-duration / packet-rate
features surfacing for DoS/DDoS)? This is the evidence that federated explanations are
*meaningful*, not just present.
"""
import numpy as np


def top_k_features(mean_abs_shap: list[float], feature_names: list[str], k: int = 5) -> list[str]:
    order = np.argsort(mean_abs_shap)[::-1][:k]
    return [feature_names[i] for i in order]


def alignment_table(global_summary: dict, feature_names: list[str], signature_features: dict, top_k: int = 5) -> list[dict]:
    rows = []
    for attack_class, known_features in signature_features.items():
        if attack_class not in global_summary:
            continue
        top_shap = top_k_features(global_summary[attack_class]["mean_abs_shap"], feature_names, k=top_k)
        overlap = sorted(set(top_shap) & set(known_features))
        rows.append({
            "attack_class": attack_class,
            "top_shap_features": top_shap,
            "known_signature_features": known_features,
            "overlap_features": overlap,
            "overlap_count": len(overlap),
            "overlap_fraction_of_known": len(overlap) / len(known_features) if known_features else 0.0,
            "n_samples": global_summary[attack_class]["n_samples"],
        })
    return rows
