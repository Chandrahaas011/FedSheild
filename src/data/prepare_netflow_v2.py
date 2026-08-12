"""Stream and canonicalize compatible UQ NetFlow-v2 datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml


LABEL_COLUMNS = {"Label", "Attack"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_csv_member(archive: ZipFile) -> str:
    members = [
        item for item in archive.namelist()
        if item.lower().endswith(".csv") and "features" not in item.lower()
    ]
    if len(members) != 1:
        raise ValueError(f"expected one data CSV in archive, found {members}")
    return members[0]


def canonicalize_archive(
    archive_path: Path,
    dataset_key: str,
    output_path: Path,
    exclude_name_patterns: list[str],
    maximum_rows_per_class: int,
    chunk_rows: int,
) -> dict:
    """Keep globally smallest deterministic feature hashes within each class."""
    excluded = {value.upper() for value in exclude_name_patterns}
    retained: dict[int, pd.DataFrame] = {0: pd.DataFrame(), 1: pd.DataFrame()}
    scanned = invalid = chunk_duplicates = 0
    with ZipFile(archive_path) as archive:
        member = archive_csv_member(archive)
        with archive.open(member) as handle:
            reader = pd.read_csv(handle, chunksize=chunk_rows, low_memory=False)
            feature_columns: list[str] | None = None
            for chunk in reader:
                scanned += len(chunk)
                if feature_columns is None:
                    feature_columns = [
                        column for column in chunk.columns
                        if column not in LABEL_COLUMNS and column.upper() not in excluded
                    ]
                    if not feature_columns:
                        raise ValueError("no model features remain after exclusions")
                labels = pd.to_numeric(chunk["Label"], errors="coerce")
                features = chunk[feature_columns].apply(pd.to_numeric, errors="coerce")
                matrix = features.to_numpy(dtype=np.float64)
                valid = np.isfinite(matrix).all(axis=1) & labels.isin([0, 1]).to_numpy()
                invalid += int((~valid).sum())
                if not valid.any():
                    continue
                candidate = features.loc[valid].astype(np.float64).reset_index(drop=True)
                candidate["binary_label"] = labels.loc[valid].astype(np.int8).to_numpy()
                candidate["family_label"] = (
                    chunk.loc[valid, "Attack"].fillna("Unknown").astype(str).to_numpy()
                )
                fingerprints = pd.util.hash_pandas_object(
                    candidate[feature_columns], index=False
                ).to_numpy(dtype=np.uint64)
                candidate["feature_fingerprint"] = fingerprints
                candidate["sample_priority"] = fingerprints ^ np.uint64(0x9E3779B97F4A7C15)
                before = len(candidate)
                candidate = candidate.drop_duplicates(
                    subset=["feature_fingerprint", "binary_label"], keep="first"
                )
                chunk_duplicates += before - len(candidate)
                for label in (0, 1):
                    part = candidate[candidate["binary_label"] == label]
                    if part.empty:
                        continue
                    combined = pd.concat([retained[label], part], ignore_index=True)
                    combined = combined.drop_duplicates(
                        subset=["feature_fingerprint"], keep="first"
                    )
                    retained[label] = combined.nsmallest(
                        maximum_rows_per_class, "sample_priority", keep="first"
                    ).reset_index(drop=True)
    if feature_columns is None:
        raise ValueError("archive contained no data rows")
    canonical = pd.concat(retained.values(), ignore_index=True)
    contradictory = canonical.groupby("feature_fingerprint")["binary_label"].nunique()
    contradictory_hashes = set(contradictory[contradictory > 1].index.tolist())
    contradictory_rows = int(canonical["feature_fingerprint"].isin(contradictory_hashes).sum())
    canonical = canonical[~canonical["feature_fingerprint"].isin(contradictory_hashes)].copy()
    canonical = canonical.drop_duplicates(subset=["feature_fingerprint"], keep="first")
    canonical["source_dataset"] = dataset_key
    canonical = canonical.sort_values(
        ["binary_label", "sample_priority", "feature_fingerprint"]
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output_path, index=False)
    return {
        "dataset_key": dataset_key,
        "archive_path": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "archive_bytes": int(archive_path.stat().st_size),
        "rows_scanned": int(scanned),
        "invalid_rows": int(invalid),
        "within_chunk_duplicates_removed": int(chunk_duplicates),
        "candidate_contradictory_fingerprints": int(len(contradictory_hashes)),
        "candidate_contradictory_rows_removed": contradictory_rows,
        "canonical_rows": int(len(canonical)),
        "class_counts": {
            str(label): int((canonical["binary_label"] == label).sum())
            for label in (0, 1)
        },
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "excluded_columns": sorted(LABEL_COLUMNS | excluded),
        "sampling": "globally smallest 64-bit feature hashes per binary class",
        "maximum_rows_per_class": int(maximum_rows_per_class),
        "limitation": (
            "Contradiction and duplicate counts are exact within the bounded hash-priority "
            "candidate set, not a full-dataset duplicate census."
        ),
    }


def prepare(config: dict, force: bool = False) -> dict:
    output_dir = Path(config["processed_dir"])
    metadata_path = Path(config["metadata_path"])
    if metadata_path.exists() and not force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "dataset_version": config["dataset_version"],
        "source_page": "https://staff.itee.uq.edu.au/marius/NIDS_datasets/",
        "license_note": (
            "The official UQ data record grants perpetual academic research use when the "
            "source papers are cited; commercial use requires author agreement. No broader "
            "redistribution license was inferred, so raw archives are not redistributed."
        ),
        "datasets": {},
    }
    for key, record in config["source_records"].items():
        metadata["datasets"][key] = canonicalize_archive(
            Path(record["archive_path"]),
            key,
            output_dir / f"{key}.parquet",
            config["exclude_name_patterns"],
            int(config["maximum_rows_per_class"]),
            int(config["chunk_rows"]),
        )
        metadata["datasets"][key]["published_rows"] = int(record["published_rows"])
        metadata["datasets"][key]["source_uuid"] = record["uuid"]
    feature_sets = {
        tuple(record["feature_columns"]) for record in metadata["datasets"].values()
    }
    if len(feature_sets) != 1:
        raise AssertionError("NetFlow-v2 canonical feature schemas do not match")
    metadata["common_feature_columns"] = list(next(iter(feature_sets)))
    metadata["common_feature_count"] = len(metadata["common_feature_columns"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/netflow_v2_cross.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    metadata = prepare(config, force=args.force)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
