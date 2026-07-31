import numpy as np

from src.data.partition import client_class_counts, partition_data


def _make_labels(n_classes=4, n_per_class=200):
    return np.concatenate([np.full(n_per_class, c) for c in range(n_classes)])


def test_partition_covers_all_indices_no_overlap():
    y = _make_labels()
    client_indices = partition_data(y, n_clients=6, alpha=0.5, seed=0)
    all_idx = np.concatenate(client_indices)
    assert len(all_idx) == len(y)
    assert len(set(all_idx.tolist())) == len(y)


def test_iid_partition_balanced_across_clients():
    y = _make_labels(n_classes=3, n_per_class=300)
    client_indices = partition_data(y, n_clients=3, alpha="iid", seed=0)
    counts = client_class_counts(client_indices, y, n_classes=3)
    proportions = counts / counts.sum(axis=1, keepdims=True)
    assert np.allclose(proportions, 1 / 3, atol=0.05)


def test_low_alpha_more_skewed_than_high_alpha():
    y = _make_labels(n_classes=5, n_per_class=400)
    low_alpha = partition_data(y, n_clients=8, alpha=0.05, seed=1)
    high_alpha = partition_data(y, n_clients=8, alpha=10.0, seed=1)

    def avg_max_class_share(client_indices):
        counts = client_class_counts(client_indices, y, n_classes=5)
        proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        return proportions.max(axis=1).mean()

    assert avg_max_class_share(low_alpha) > avg_max_class_share(high_alpha)


def test_reproducible_with_same_seed():
    y = _make_labels()
    a = partition_data(y, n_clients=4, alpha=0.5, seed=42)
    b = partition_data(y, n_clients=4, alpha=0.5, seed=42)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca, cb)
