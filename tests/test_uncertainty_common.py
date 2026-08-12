import numpy as np

from experiments.uncertainty_common import evaluate_uncertainty, validation_roles
from experiments.density_uncertainty_common import evaluate_density_uncertainty


def test_validation_roles_are_disjoint_exhaustive_and_deterministic():
    labels = np.array([0, 1] * 100)
    first = validation_roles(labels, seed=7)
    second = validation_roles(labels, seed=7)
    for key in first:
        assert np.array_equal(first[key], second[key])
    combined = np.concatenate(list(first.values()))
    assert len(combined) == len(labels)
    assert len(np.unique(combined)) == len(labels)
    assert set(combined) == set(range(len(labels)))


def test_uncertainty_evaluation_contains_frozen_primary_method():
    y_cal = np.array([0] * 50 + [1] * 50)
    p_cal = np.linspace(0.01, 0.99, 100)
    output = evaluate_uncertainty(
        y_cal, p_cal, y_cal, p_cal, alpha=0.10, target_coverage=0.90
    )
    assert "mondrian_conformal_alpha_10" in output["methods"]
    assert len(output["curves"]["confidence"]) == 8
    assert len(output["curves"]["mondrian_conformal"]) == 7


def test_density_uncertainty_contains_frozen_fused_method():
    y = np.array([0] * 50 + [1] * 50)
    probability = np.linspace(0.01, 0.99, 100)
    novelty = np.column_stack([probability, 1.0 - probability])
    output = evaluate_density_uncertainty(
        y,
        probability,
        novelty,
        y,
        probability,
        novelty,
        alpha=0.10,
        target_coverage=0.90,
        probability_weight=0.50,
    )
    assert "distance_mondrian_alpha_10" in output["methods"]
    assert "fused_distance_probability_mondrian_alpha_10" in output["methods"]
    assert len(output["curves"]["fused_mondrian"]) == 7
