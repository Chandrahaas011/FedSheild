import numpy as np

from src.fl.sparsify import residual_adaptive_topk_sparsify


def test_residual_adaptive_topk_respects_bounds_and_energy():
    delta = np.array([10.0, 3.0, 1.0, 0.1, 0.1], dtype=np.float32)
    idx, values, residual, ratio = residual_adaptive_topk_sparsify(
        delta, np.zeros_like(delta), min_ratio=0.2, max_ratio=0.8, energy_target=0.9
    )
    assert 0.2 <= ratio <= 0.8
    assert len(idx) == len(values)
    assert np.all(residual[idx] == 0)
    sent_energy = float(np.square(values).sum())
    assert sent_energy / float(np.square(delta).sum()) >= 0.9


def test_residual_adaptive_topk_keeps_error_feedback():
    delta = np.array([1.0, 0.5, 0.1, 0.05], dtype=np.float32)
    idx, _, residual, _ = residual_adaptive_topk_sparsify(
        delta, np.zeros_like(delta), min_ratio=0.25, max_ratio=0.25, energy_target=0.5
    )
    unsent = np.setdiff1d(np.arange(len(delta)), idx)
    assert np.allclose(residual[unsent], delta[unsent])
