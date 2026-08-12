import numpy as np
import pytest

from src.evaluation.statistics import paired_effect_summary, risk_coverage_area


def test_paired_effect_summary_uses_candidate_minus_baseline():
    output = paired_effect_summary(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 1.5, 2.5]),
        bootstrap_repeats=1000,
        seed=7,
    )
    assert output["mean_difference"] == -0.5
    assert output["candidate_lower_count"] == 3
    assert output["candidate_higher_count"] == 0
    assert output["exact_sign_flip_p_two_sided"] == 0.25


def test_risk_coverage_area_is_normalized_by_span():
    output = risk_coverage_area(
        np.array([0.5, 0.75, 1.0]), np.array([0.0, 0.1, 0.2])
    )
    assert output["risk_coverage_area"] == pytest.approx(0.05)
    assert output["mean_risk_over_coverage_span"] == pytest.approx(0.1)


def test_risk_coverage_requires_distinct_coverage():
    with pytest.raises(ValueError, match="distinct"):
        risk_coverage_area(np.array([0.5, 0.5]), np.array([0.1, 0.2]))
