from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split


def _stratified_cap(indices: np.ndarray, labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    selected, _ = train_test_split(
        indices, train_size=cap, random_state=seed, stratify=labels[indices]
    )
    return np.asarray(selected)


def campaign_stable_feature_ranking(
    X: np.ndarray,
    y: np.ndarray,
    environments: np.ndarray,
    feature_names: list[str],
    seed: int,
    repeats: int = 2,
    sample_rows_per_environment: int = 50000,
) -> pd.DataFrame:
    """Rank features by conservative importance across attack-bearing capture days.

    Only training rows are accepted by the caller. Environments lacking both classes are
    skipped. The score is the mean normalized tree importance minus half its between-fit
    standard deviation, clipped at zero. It rewards useful features whose importance is
    less dependent on one attack campaign.
    """
    importance_vectors = []
    used_environments = []
    for environment in sorted(np.unique(environments).tolist()):
        environment_indices = np.flatnonzero(environments == environment)
        if len(np.unique(y[environment_indices])) < 2 or y[environment_indices].sum() < 20:
            continue
        for repeat in range(repeats):
            fit_seed = seed + 1009 * repeat + 37 * len(used_environments)
            selected = _stratified_cap(
                environment_indices, y, sample_rows_per_environment, fit_seed
            )
            model = ExtraTreesClassifier(
                n_estimators=48,
                max_depth=14,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=fit_seed,
            )
            model.fit(X[selected], y[selected])
            importance = model.feature_importances_.astype(float)
            importance /= max(importance.sum(), np.finfo(float).eps)
            importance_vectors.append(importance)
            used_environments.append(str(environment))
    if not importance_vectors:
        raise ValueError("No training environment contains both classes")

    matrix = np.vstack(importance_vectors)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    score = np.maximum(mean - 0.5 * std, 0.0)
    ranking = pd.DataFrame(
        {
            "feature": feature_names,
            "stability_score": score,
            "importance_mean": mean,
            "importance_std": std,
        }
    ).sort_values(["stability_score", "importance_mean", "feature"], ascending=[False, False, True])
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.attrs["training_environments"] = sorted(set(used_environments))
    return ranking.reset_index(drop=True)

