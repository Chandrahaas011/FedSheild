"""Preprocessing pipeline for the corrected CICIDS2017 dataset (Engelen et al., WTMC 2021).

Produces a clean, scaled, split, parquet dataset from the raw per-day CSVs:
    data/raw/cicids2017_corrected/*.csv
      -> data/processed/{train,val,test}.parquet
      -> data/processed/metadata.json
      -> data/processed/scaler.npz
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.labels import CICIDS2017_CLASSES, CICIDS2017_DROP_COLUMNS, CICIDS2017_TAXONOMY

RAW_DIR = Path("data/raw/cicids2017_corrected")
OUT_DIR = Path("data/processed")
DAY_FILES = [
    "Monday-WorkingHours.csv",
    "Tuesday-WorkingHours.csv",
    "Wednesday-WorkingHours.csv",
    "Thursday-WorkingHours.csv",
    "Friday-WorkingHours.csv",
]


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = []
    for fname in DAY_FILES:
        path = raw_dir / fname
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_and_label(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_raw": int(len(df))}

    df["Label"] = df["Label"].str.strip()
    unmapped = sorted(set(df["Label"].unique()) - set(CICIDS2017_TAXONOMY.keys()))
    if unmapped:
        raise ValueError(f"Unmapped raw labels found, update labels.py: {unmapped}")
    df["attack_class"] = df["Label"].map(CICIDS2017_TAXONOMY)

    feature_cols = [c for c in df.columns if c not in CICIDS2017_DROP_COLUMNS and c != "attack_class"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    stats["rows_dropped_inf_nan"] = int(n_before - len(df))

    n_before = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["attack_class"])
    stats["rows_dropped_duplicates"] = int(n_before - len(df))

    stats["rows_clean"] = int(len(df))
    stats["class_counts_clean"] = df["attack_class"].value_counts().to_dict()
    return df, stats


def split_scale(df: pd.DataFrame, seed: int = 42):
    feature_cols = [c for c in df.columns if c not in CICIDS2017_DROP_COLUMNS and c != "attack_class"]
    class_to_idx = {c: i for i, c in enumerate(CICIDS2017_CLASSES)}
    y = df["attack_class"].map(class_to_idx).to_numpy(dtype=np.int64)
    X = df[feature_cols].to_numpy(dtype=np.float32)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    return splits, scaler, feature_cols, class_to_idx


def save_splits(splits: dict, feature_cols: list[str], out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (X, y) in splits.items():
        df = pd.DataFrame(X, columns=feature_cols)
        df["label"] = y
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def run(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR, seed: int = 42) -> dict:
    df = load_raw(raw_dir)
    df, stats = clean_and_label(df)
    splits, scaler, feature_cols, class_to_idx = split_scale(df, seed=seed)
    save_splits(splits, feature_cols, out_dir)

    np.savez(out_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    metadata = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "classes": CICIDS2017_CLASSES,
        "class_to_idx": class_to_idx,
        "split_sizes": {k: int(len(v[1])) for k, v in splits.items()},
        "split_class_counts": {
            k: {CICIDS2017_CLASSES[c]: int((v[1] == c).sum()) for c in np.unique(v[1])}
            for k, v in splits.items()
        },
        "seed": seed,
        "stats": stats,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


if __name__ == "__main__":
    meta = run()
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_columns"}, indent=2, default=str))
