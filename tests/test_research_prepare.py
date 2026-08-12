import numpy as np
import pandas as pd

from src.data.research_prepare import (
    deterministic_stratified_split,
    remove_ambiguous_and_duplicate_features,
)


def test_conflicting_feature_groups_are_removed_without_relabeling():
    frame = pd.DataFrame(
        {
            "f1": [1.0, 1.0, 2.0, 2.0, 3.0],
            "f2": [5.0, 5.0, 6.0, 6.0, 7.0],
            "attack_class": ["Benign", "PortScan", "DoS", "DoS", "Benign"],
        }
    )
    clean, stats = remove_ambiguous_and_duplicate_features(frame, ["f1", "f2"])
    assert len(clean) == 2
    assert set(clean["attack_class"]) == {"DoS", "Benign"}
    assert stats["ambiguous_rows_removed"] == 2
    assert stats["same_label_duplicate_rows_removed"] == 1


def test_research_split_is_deterministic_disjoint_and_complete():
    labels = np.repeat(np.arange(3), 100)
    first = deterministic_stratified_split(labels, seed=7)
    second = deterministic_stratified_split(labels, seed=7)
    assert np.array_equal(first, second)
    assert set(first) == {"train", "validation", "test"}
    assert len(first) == len(labels)
    assert np.all(np.array([np.sum(first == name) for name in set(first)]) > 0)

