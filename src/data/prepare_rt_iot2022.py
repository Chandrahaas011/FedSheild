"""Prepare a leakage-audited RT-IoT2022 replication dataset.

The public file contains an exported row index, two endpoint ports and two
categorical protocol/service fields.  This conservative replication excludes
all five because endpoint/service identifiers can encode collection artifacts;
the remaining flow statistics are numeric and directly usable by every
lightweight baseline in the study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


BENIGN_LABELS = {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}
EXCLUDED_COLUMNS = {"Attack_type", "id.orig_p", "id.resp_p", "proto", "service"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(config: dict) -> dict:
    raw_path = Path(config["raw_path"])
    output_path = Path(config["dataset_path"])
    metadata_path = Path(config["metadata_path"])
    frame = pd.read_csv(raw_path)
    unnamed = [column for column in frame if column.startswith("Unnamed:") or column == ""]
    frame = frame.drop(columns=unnamed)
    feature_columns = [column for column in frame if column not in EXCLUDED_COLUMNS]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    removed_nonfinite = int((~finite_mask).sum())
    frame = frame.loc[finite_mask].reset_index(drop=True)
    numeric = numeric.loc[finite_mask].reset_index(drop=True).astype(np.float32)

    labels = frame["Attack_type"].astype(str)
    binary = (~labels.isin(BENIGN_LABELS)).astype(np.int8)
    hashes = pd.util.hash_pandas_object(numeric, index=False).astype(str)
    audit = pd.DataFrame({"fingerprint": hashes, "attack_family": labels, "binary_label": binary})
    ambiguity = audit.groupby("fingerprint")["binary_label"].nunique()
    ambiguous_hashes = set(ambiguity[ambiguity > 1].index)
    keep = ~hashes.isin(ambiguous_hashes)
    removed_ambiguous_rows = int((~keep).sum())

    clean = numeric.loc[keep].copy()
    clean["attack_family"] = labels.loc[keep].to_numpy()
    clean["binary_label"] = binary.loc[keep].to_numpy()
    clean["feature_fingerprint"] = hashes.loc[keep].to_numpy()
    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset=["feature_fingerprint", "binary_label"], keep="first")
    removed_same_label_duplicates = before_dedup - len(clean)

    indices = np.arange(len(clean))
    development, test = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=clean["attack_family"]
    )
    train, validation = train_test_split(
        development,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=clean.iloc[development]["attack_family"],
    )
    split = np.empty(len(clean), dtype=object)
    split[train], split[validation], split[test] = "train", "validation", "test"
    clean["split_random"] = split

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(output_path, index=False)
    split_counts = {}
    for name in ("train", "validation", "test"):
        part = clean[clean["split_random"] == name]
        split_counts[name] = {
            "rows": int(len(part)),
            "binary_counts": {str(k): int(v) for k, v in part["binary_label"].value_counts().sort_index().items()},
            "family_counts": {str(k): int(v) for k, v in part["attack_family"].value_counts().sort_index().items()},
        }
    metadata = {
        "dataset": "RT-IoT2022",
        "source": "UCI Machine Learning Repository, DOI 10.24432/C5P338",
        "license": "CC BY 4.0",
        "raw_sha256": sha256(raw_path),
        "rows_raw": int(len(frame) + removed_nonfinite),
        "rows_clean": int(len(clean)),
        "removed_nonfinite": removed_nonfinite,
        "ambiguous_fingerprints": int(len(ambiguous_hashes)),
        "removed_ambiguous_rows": removed_ambiguous_rows,
        "removed_same_label_duplicates": int(removed_same_label_duplicates),
        "feature_columns": feature_columns,
        "excluded_columns": sorted(EXCLUDED_COLUMNS | set(unnamed)),
        "benign_labels": sorted(BENIGN_LABELS),
        "split_seed": 42,
        "split_counts": split_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rt_iot2022.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(prepare(config), indent=2))


if __name__ == "__main__":
    main()
