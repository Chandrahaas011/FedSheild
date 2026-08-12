"""Leakage-aware preparation for the corrected CICIDS2017 research protocol.

The legacy pipeline removed duplicates using (features, target), which retained exact
feature vectors carrying contradictory targets. This module keeps capture provenance,
quarantines every contradictory feature fingerprint, removes remaining exact duplicate
feature vectors, and creates deterministic 70/15/15 stratified partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.data.labels import (
    CICIDS2017_CLASSES,
    CICIDS2017_DROP_COLUMNS,
    CICIDS2017_TAXONOMY,
)

RAW_DIR = Path("data/raw/cicids2017_corrected")
OUTPUT_PATH = Path("data/processed/research_cicids2017.parquet")
METADATA_PATH = Path("data/processed/research_metadata.json")
DAY_FILES = [
    "Monday-WorkingHours.csv",
    "Tuesday-WorkingHours.csv",
    "Wednesday-WorkingHours.csv",
    "Thursday-WorkingHours.csv",
    "Friday-WorkingHours.csv",
]
PROVENANCE_COLUMNS = [
    "capture_day",
    "source_file",
    "source_row",
    "timestamp_raw",
    "raw_label",
    "attack_class",
    "binary_label",
    "feature_fingerprint",
    "split_random",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def feature_fingerprints(features: pd.DataFrame) -> pd.Series:
    """Return a deterministic uint64 fingerprint for each exact feature row."""
    return pd.util.hash_pandas_object(features, index=False).astype("uint64")


def remove_ambiguous_and_duplicate_features(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Remove feature vectors with conflicting families, then exact duplicates.

    Ambiguous groups are removed in full: selecting one target would silently rewrite
    labels. Once ambiguity is removed, one representative of each exact feature vector
    is retained so frequency artifacts cannot dominate evaluation.
    """
    work = frame.copy()
    work["feature_fingerprint"] = feature_fingerprints(work[feature_columns]).to_numpy()
    group_label_counts = work.groupby("feature_fingerprint", sort=False)["attack_class"].nunique()
    ambiguous = group_label_counts[group_label_counts > 1].index
    ambiguous_mask = work["feature_fingerprint"].isin(ambiguous)

    ambiguity_pairs = (
        work.loc[ambiguous_mask]
        .groupby("feature_fingerprint")["attack_class"]
        .agg(lambda values: " | ".join(sorted(set(values))))
        .value_counts()
        .to_dict()
    )
    clean = work.loc[~ambiguous_mask].copy()
    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset=feature_columns, keep="first").reset_index(drop=True)

    stats = {
        "ambiguous_fingerprints_removed": int(len(ambiguous)),
        "ambiguous_rows_removed": int(ambiguous_mask.sum()),
        "ambiguous_class_pairs": {str(k): int(v) for k, v in ambiguity_pairs.items()},
        "same_label_duplicate_rows_removed": int(before_dedup - len(clean)),
    }
    return clean, stats


def deterministic_stratified_split(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Create deterministic 70/15/15 split labels for unique feature rows."""
    indices = np.arange(len(labels))
    train_idx, temporary_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=labels
    )
    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=0.50,
        random_state=seed,
        stratify=labels[temporary_idx],
    )
    split = np.empty(len(labels), dtype="U10")
    split[train_idx] = "train"
    split[validation_idx] = "validation"
    split[test_idx] = "test"
    return split


def _load_one_day(path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [column.strip() for column in frame.columns]
    raw_rows = len(frame)
    frame["Label"] = frame["Label"].astype(str).str.strip()
    unmapped = sorted(set(frame["Label"].unique()) - set(CICIDS2017_TAXONOMY))
    if unmapped:
        raise ValueError(f"Unmapped labels in {path.name}: {unmapped}")

    feature_columns = [
        column
        for column in frame.columns
        if column not in CICIDS2017_DROP_COLUMNS and column != "attack_class"
    ]
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_mask = np.isfinite(frame[feature_columns].to_numpy(dtype=np.float64)).all(axis=1)

    timestamp = frame["Timestamp"].astype(str) if "Timestamp" in frame else pd.Series("", index=frame.index)
    output = frame.loc[finite_mask, feature_columns].copy()
    output["capture_day"] = path.stem.split("-")[0]
    output["source_file"] = path.name
    output["source_row"] = frame.index[finite_mask].to_numpy(dtype=np.int64)
    output["timestamp_raw"] = timestamp.loc[finite_mask].to_numpy()
    output["raw_label"] = frame.loc[finite_mask, "Label"].to_numpy()
    output["attack_class"] = output["raw_label"].map(CICIDS2017_TAXONOMY)
    output["binary_label"] = (output["attack_class"] != "Benign").astype("int8")
    stats = {
        "rows_raw": int(raw_rows),
        "rows_nonfinite_removed": int(raw_rows - finite_mask.sum()),
        "rows_finite": int(finite_mask.sum()),
        "family_counts_finite": {
            str(k): int(v) for k, v in output["attack_class"].value_counts().items()
        },
    }
    return output, feature_columns, stats


def prepare(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
    seed: int = 42,
    force: bool = False,
) -> dict:
    if output_path.exists() and metadata_path.exists() and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    frames: list[pd.DataFrame] = []
    day_stats: dict[str, dict] = {}
    feature_columns: list[str] | None = None
    for file_name in DAY_FILES:
        day_frame, current_features, current_stats = _load_one_day(raw_dir / file_name)
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(f"Feature schema mismatch in {file_name}")
        frames.append(day_frame)
        day_stats[file_name] = current_stats

    assert feature_columns is not None
    combined = pd.concat(frames, ignore_index=True)
    cleaned, conflict_stats = remove_ambiguous_and_duplicate_features(combined, feature_columns)
    del combined, frames

    family_index = {name: index for index, name in enumerate(CICIDS2017_CLASSES)}
    cleaned["family_label"] = cleaned["attack_class"].map(family_index).astype("int8")
    cleaned["split_random"] = deterministic_stratified_split(
        cleaned["family_label"].to_numpy(), seed=seed
    )

    ordered_columns = feature_columns + [
        "capture_day",
        "source_file",
        "source_row",
        "timestamp_raw",
        "raw_label",
        "attack_class",
        "family_label",
        "binary_label",
        "feature_fingerprint",
        "split_random",
    ]
    cleaned = cleaned[ordered_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False, compression="zstd")

    split_counts = {}
    for split_name, split_frame in cleaned.groupby("split_random"):
        split_counts[str(split_name)] = {
            "rows": int(len(split_frame)),
            "binary_counts": {
                str(k): int(v) for k, v in split_frame["binary_label"].value_counts().items()
            },
            "family_counts": {
                str(k): int(v) for k, v in split_frame["attack_class"].value_counts().items()
            },
        }

    archive_path = raw_dir.parent / "cicids2017_corrected.zip"
    metadata = {
        "dataset": "corrected_cicids2017",
        "output_path": str(output_path),
        "raw_archive_sha256": sha256_file(archive_path) if archive_path.exists() else None,
        "seed": seed,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "rows_final": int(len(cleaned)),
        "classes": CICIDS2017_CLASSES,
        "day_stats": day_stats,
        "cleaning": conflict_stats,
        "split_counts": split_counts,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol_notes": [
            "All contradictory exact feature fingerprints are removed in full.",
            "Remaining exact feature duplicates are reduced to one representative.",
            "Random partitions are stratified by attack family after cleaning.",
            "Features are unscaled; each experiment fits preprocessing on training only.",
            "Capture day and raw source provenance are retained for shift evaluation.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metadata = prepare(args.raw_dir, args.output, args.metadata, args.seed, args.force)
    print(json.dumps({k: v for k, v in metadata.items() if k != "feature_columns"}, indent=2))


if __name__ == "__main__":
    main()

