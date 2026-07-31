import numpy as np

from src.fl.sparsify import densify, sparse_bytes, topk_sparsify


def test_topk_selects_largest_magnitude():
    delta = np.array([0.1, -5.0, 0.2, 3.0, -0.05, 0.01], dtype=np.float32)
    residual = np.zeros_like(delta)
    idx, values, new_residual = topk_sparsify(delta, ratio=1 / 3, residual=residual)
    assert set(idx.tolist()) == {1, 3}
    assert np.allclose(sorted(values), sorted([-5.0, 3.0]), atol=1e-5)


def test_residual_carries_unsent_mass():
    delta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    residual = np.zeros_like(delta)
    idx, values, new_residual = topk_sparsify(delta, ratio=0.25, residual=residual)
    assert new_residual[idx].sum() == 0
    untouched = np.setdiff1d(np.arange(4), idx)
    assert np.allclose(new_residual[untouched], delta[untouched])


def test_residual_accumulates_across_rounds_until_sent():
    # A coordinate that's never top-1 individually should still eventually get sent
    # once its accumulated residual makes it large enough.
    residual = np.zeros(3, dtype=np.float32)
    small_delta = np.array([0.05, 10.0, 0.05], dtype=np.float32)
    _, _, residual = topk_sparsify(small_delta, ratio=1 / 3, residual=residual)
    assert np.isclose(residual[0], 0.05) and np.isclose(residual[2], 0.05)

    idx2, values2, _ = topk_sparsify(np.array([0.05, 0.0, 0.05], dtype=np.float32), ratio=1 / 3, residual=residual)
    # combined residual+delta for coord 0 and 2 is now 0.10 each, still less than nothing
    # sent for coord1 (0), so top-1 of [0.1, 0.0, 0.1] picks either coord 0 or 2.
    assert idx2[0] in (0, 2)


def test_densify_inverts_sparse_selection():
    idx = np.array([1, 3])
    values = np.array([2.0, -1.0], dtype=np.float32)
    dense = densify(idx, values, length=5)
    expected = np.array([0, 2.0, 0, -1.0, 0], dtype=np.float32)
    assert np.allclose(dense, expected)


def test_sparse_bytes_smaller_than_dense_for_small_ratio():
    length = 1000
    delta = np.random.randn(length).astype(np.float32)
    idx, values, _ = topk_sparsify(delta, ratio=0.1, residual=np.zeros(length, dtype=np.float32))
    assert sparse_bytes(idx, values) < delta.nbytes
