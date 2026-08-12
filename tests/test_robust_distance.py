import numpy as np
import pytest

from src.models.robust_distance import ClassConditionalRobustDistance


def test_robust_distance_is_deterministic_and_class_conditional():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.2], [-0.1, 0.1], [10.0, 10.0], [10.2, 9.9], [9.8, 10.1]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    first = ClassConditionalRobustDistance(100, seed=3).fit(X, y)
    second = ClassConditionalRobustDistance(100, seed=3).fit(X, y)
    scores = first.candidate_novelty(np.array([[0.05, 0.05], [10.1, 10.0]], dtype=np.float32))
    assert np.array_equal(scores, second.candidate_novelty(np.array([[0.05, 0.05], [10.1, 10.0]], dtype=np.float32)))
    assert scores[0, 0] < scores[0, 1]
    assert scores[1, 1] < scores[1, 0]


def test_robust_distance_rejects_wrong_feature_count():
    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    model = ClassConditionalRobustDistance(100).fit(X, np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="feature count"):
        model.candidate_novelty(np.zeros((1, 2), dtype=np.float32))
