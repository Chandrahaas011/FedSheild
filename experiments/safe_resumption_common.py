"""Leakage-safe role allocation and gate logic for bounded recovery experiments."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def allocate_recovery_roles(
    target: pd.DataFrame,
    maximum_review_rows: int,
    validation_rows: int,
    audit_rows: int,
    seed: int,
    shadow_validation_rows: int = 0,
) -> dict[str, pd.DataFrame]:
    """Create disjoint review, gate-validation, and final-audit target roles.

    The review pool is balanced and nested by construction. Validation and audit retain
    the observed target prevalence through stratified sampling from all remaining rows.
    """
    labels = target["binary_label"].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("target must contain both binary classes")
    per_class = maximum_review_rows // 2
    rng = np.random.default_rng(seed + 11001)
    review_indices = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"target has fewer than {per_class} rows for class {label}")
        review_indices.extend(rng.permutation(candidates)[:per_class].tolist())
    review_indices = np.asarray(review_indices, dtype=int)
    remaining = np.setdiff1d(np.arange(len(target)), review_indices, assume_unique=False)
    validation_count = min(int(validation_rows), len(remaining) - 2)
    validation_indices, remaining = train_test_split(
        remaining,
        train_size=validation_count,
        random_state=seed + 11011,
        stratify=labels[remaining],
    )
    shadow_indices = np.asarray([], dtype=int)
    if shadow_validation_rows:
        shadow_count = min(int(shadow_validation_rows), len(remaining) - 2)
        shadow_indices, remaining = train_test_split(
            remaining,
            train_size=shadow_count,
            random_state=seed + 11017,
            stratify=labels[remaining],
        )
    audit_count = min(int(audit_rows), len(remaining))
    if audit_count < 2:
        raise ValueError("insufficient rows for final recovery audit")
    if audit_count < len(remaining):
        audit_indices, _ = train_test_split(
            remaining,
            train_size=audit_count,
            random_state=seed + 11021,
            stratify=labels[remaining],
        )
    else:
        audit_indices = remaining
    roles = {
        "review": target.iloc[np.sort(review_indices)].copy(),
        "validation": target.iloc[np.sort(validation_indices)].copy(),
    }
    if len(shadow_indices):
        roles["shadow_validation"] = target.iloc[np.sort(shadow_indices)].copy()
    roles["audit"] = target.iloc[np.sort(audit_indices)].copy()
    for frame in roles.values():
        frame["recovery_original_index"] = frame.index.to_numpy(dtype=int)
        frame.reset_index(drop=True, inplace=True)
    return roles


def nested_balanced_review(review: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    """Return a deterministic balanced subset; smaller budgets nest within larger ones."""
    if budget <= 0 or budget % 2:
        raise ValueError("review budget must be a positive even integer")
    per_class = budget // 2
    pieces = []
    for label in (0, 1):
        group = review[review["binary_label"] == label]
        if len(group) < per_class:
            raise ValueError("review pool cannot satisfy balanced label budget")
        order = np.random.default_rng(seed + 11101 + label).permutation(len(group))
        pieces.append(group.iloc[np.sort(order[:per_class])])
    return pd.concat(pieces).sort_values("recovery_original_index").reset_index(drop=True)


def attach_review_label_noise(
    review: pd.DataFrame,
    noise_rate: float,
    seed: int,
    rates_by_clean_label: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Attach deterministic symmetric or class-conditional label corruption.

    The clean ``binary_label`` is preserved for auditability. Corruption is assigned on
    the full review pool before nested budgets are drawn, so a row's noisy label remains
    stable wherever that row appears.
    """
    if not 0.0 <= noise_rate < 0.5:
        raise ValueError("review-label noise rate must satisfy 0 <= rate < 0.5")
    rates = rates_by_clean_label or {0: noise_rate, 1: noise_rate}
    if set(rates) != {0, 1} or any(not 0.0 <= float(rate) < 0.5 for rate in rates.values()):
        raise ValueError("class-conditional rates must define labels 0 and 1 in [0, 0.5)")
    output = review.copy()
    output["recovery_training_label"] = output["binary_label"].astype(np.int8)
    output["review_label_corrupted"] = False
    for label in (0, 1):
        candidates = np.flatnonzero(output["binary_label"].to_numpy() == label)
        flip_count = int(np.floor(float(rates[label]) * len(candidates)))
        if not flip_count:
            continue
        order = np.random.default_rng(seed + 12101 + label).permutation(candidates)
        selected = order[:flip_count]
        output.loc[output.index[selected], "recovery_training_label"] = 1 - label
        output.loc[output.index[selected], "review_label_corrupted"] = True
    return output


def gate_passes(metrics: dict, efficiency: dict, criteria: dict) -> bool:
    return bool(
        metrics["macro_f1"] >= criteria["minimum_macro_f1"]
        and metrics["per_class"]["Malicious"]["recall"]
        >= criteria["minimum_malicious_recall"]
        and metrics["false_positive_rate"] <= criteria["maximum_false_positive_rate"]
        and metrics["ece_15"] <= criteria["maximum_ece"]
        and efficiency["serialized_model_bytes"] <= criteria["maximum_model_bytes"]
        and efficiency["single_record_latency_ms_p95"]
        <= criteria["maximum_single_record_latency_p95_ms"]
    )


def wilson_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """Return the two-sided Wilson interval's upper endpoint.

    At 95% confidence this uses z=1.95996. The implementation intentionally supports
    only the frozen level used by the sensitivity analysis rather than silently
    approximating arbitrary confidence levels.
    """
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if confidence != 0.95:
        raise ValueError("only the frozen 0.95 confidence level is supported")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return (centre + radius) / denominator


def uncertainty_aware_gate_passes(
    metrics: dict,
    efficiency: dict,
    criteria: dict,
) -> tuple[bool, float]:
    """Apply the primary gate plus a 95% Wilson upper bound on validation FPR."""
    confusion = metrics["confusion_matrix"]
    benign_trials = int(confusion[0][0]) + int(confusion[0][1])
    false_positives = int(confusion[0][1])
    upper_fpr = wilson_upper_bound(false_positives, benign_trials)
    passed = gate_passes(metrics, efficiency, criteria) and (
        upper_fpr <= criteria["maximum_false_positive_rate"]
    )
    return bool(passed), float(upper_fpr)


def choose_resumption_action(actions: list[dict]) -> dict | None:
    """Select only from validation-passing actions, never using final-audit metrics."""
    eligible = [action for action in actions if action["gate_passed"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda action: (
            action["label_budget"],
            action["efficiency"]["serialized_model_bytes"],
            -action["validation_metrics"]["macro_f1"],
            action["action"],
        ),
    )
