from pathlib import Path

import pandas as pd

import numpy as np

from experiments.run_cross_netflow import (
    _cap_label_blind,
    load_cross_partitions,
    signed_log1p,
)


def test_cross_partitions_remove_overlap_and_keep_target_out_of_development(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    columns = {
        "F1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "binary_label": [0, 0, 0, 0, 1, 1, 1, 1],
        "family_label": ["Benign"] * 4 + ["Attack"] * 4,
        "sample_priority": list(range(8)),
        "source_dataset": ["source"] * 8,
    }
    source = pd.DataFrame({**columns, "feature_fingerprint": list(range(8))})
    target = pd.DataFrame({**columns, "feature_fingerprint": [0, 10, 11, 12, 13, 14, 15, 16]})
    source.to_parquet(processed / "nf_unsw_nb15_v2.parquet", index=False)
    target.to_parquet(processed / "nf_ton_iot_v2.parquet", index=False)
    config = {
        "processed_dir": str(processed),
        "maximum_train_rows": None,
        "maximum_validation_rows": None,
        "maximum_target_test_rows": None,
    }
    parts, overlap = load_cross_partitions(
        config, "nf_unsw_nb15_v2", "nf_ton_iot_v2", seed=0
    )
    assert overlap == 1
    assert set(parts["train"].feature_fingerprint).isdisjoint(parts["test"].feature_fingerprint)
    assert set(parts["validation"].feature_fingerprint).isdisjoint(parts["test"].feature_fingerprint)


def test_signed_log_transform_makes_large_values_float32_safe():
    transformed = signed_log1p(np.array([[-1.0e300, 0.0, 1.0e300]]))
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] < 0 < transformed[0, 2]


def test_label_blind_target_cap_does_not_change_when_labels_change():
    frame = pd.DataFrame(
        {
            "feature_fingerprint": np.arange(100),
            "binary_label": np.array([0, 1] * 50),
        }
    )
    relabelled = frame.copy()
    relabelled["binary_label"] = 1 - relabelled["binary_label"]
    first = _cap_label_blind(frame, 30, seed=9)
    second = _cap_label_blind(relabelled, 30, seed=9)
    assert first["feature_fingerprint"].tolist() == second["feature_fingerprint"].tolist()
