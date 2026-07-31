import numpy as np

from src.fl.param_utils import dense_bytes, flatten, get_shapes, unflatten


def test_flatten_unflatten_roundtrip():
    arrays = [np.random.randn(4, 3).astype(np.float32), np.random.randn(10).astype(np.float32),
              np.random.randn(2, 2, 2).astype(np.float32)]
    shapes = get_shapes(arrays)
    flat, sizes = flatten(arrays)
    restored = unflatten(flat, shapes, sizes)
    for orig, rest in zip(arrays, restored):
        assert orig.shape == rest.shape
        assert np.allclose(orig, rest, atol=1e-6)


def test_dense_bytes_matches_nbytes_sum():
    arrays = [np.zeros((5, 5), dtype=np.float32), np.zeros(10, dtype=np.float32)]
    assert dense_bytes(arrays) == arrays[0].nbytes + arrays[1].nbytes
