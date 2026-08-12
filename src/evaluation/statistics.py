"""Small-sample paired statistics for reproducible experiment reporting."""
from __future__ import annotations

import itertools

import numpy as np


def paired_effect_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    bootstrap_repeats: int = 20_000,
    seed: int = 2026,
    tie_tolerance: float = 1.0e-12,
) -> dict:
    """Summarize candidate-minus-baseline paired effects.

    The exact randomization test enumerates every sign assignment when there are at
    most 20 pairs. It tests the sharp null of exchangeable difference signs.
    """
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError("paired inputs must be equal-length one-dimensional arrays")
    if len(baseline) < 2 or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("at least two finite pairs are required")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(bootstrap_repeats, len(differences)), replace=True)
    bootstrap_means = samples.mean(axis=1)
    observed = abs(float(differences.mean()))
    if len(differences) <= 20:
        permuted = np.asarray(
            [np.mean(np.asarray(signs) * differences) for signs in itertools.product((-1.0, 1.0), repeat=len(differences))]
        )
        p_value = float(np.mean(np.abs(permuted) >= observed - 1.0e-15))
    else:
        signs = rng.choice((-1.0, 1.0), size=(bootstrap_repeats, len(differences)))
        p_value = float(np.mean(np.abs(signs * differences).mean(axis=1) >= observed))
    return {
        "n_pairs": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "sample_std_difference": float(differences.std(ddof=1)),
        "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "exact_sign_flip_p_two_sided": p_value,
        "candidate_lower_count": int((differences < -tie_tolerance).sum()),
        "tie_count": int((np.abs(differences) <= tie_tolerance).sum()),
        "candidate_higher_count": int((differences > tie_tolerance).sum()),
    }


def risk_coverage_area(coverage: np.ndarray, selective_risk: np.ndarray) -> dict:
    """Trapezoidal risk--coverage area and its coverage-range normalization."""
    coverage = np.asarray(coverage, dtype=float)
    selective_risk = np.asarray(selective_risk, dtype=float)
    if coverage.ndim != 1 or selective_risk.shape != coverage.shape:
        raise ValueError("coverage and risk must be equal-length one-dimensional arrays")
    valid = np.isfinite(coverage) & np.isfinite(selective_risk)
    coverage, selective_risk = coverage[valid], selective_risk[valid]
    if len(coverage) < 2:
        raise ValueError("at least two finite risk--coverage points are required")
    order = np.argsort(coverage)
    coverage, selective_risk = coverage[order], selective_risk[order]
    unique_coverage = np.unique(coverage)
    unique_risk = np.asarray(
        [selective_risk[coverage == value].mean() for value in unique_coverage]
    )
    if len(unique_coverage) < 2 or unique_coverage[-1] <= unique_coverage[0]:
        raise ValueError("coverage must span at least two distinct values")
    area = float(np.trapz(unique_risk, unique_coverage))
    span = float(unique_coverage[-1] - unique_coverage[0])
    return {
        "coverage_min": float(unique_coverage[0]),
        "coverage_max": float(unique_coverage[-1]),
        "coverage_span": span,
        "risk_coverage_area": area,
        "mean_risk_over_coverage_span": float(area / span),
        "n_points": int(len(unique_coverage)),
    }
