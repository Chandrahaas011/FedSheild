from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.result_schema import validate_result

PROVENANCE_COLUMNS = [
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_cap(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    counts = frame["family_label"].value_counts().sort_index()
    exact = counts * (maximum / len(frame))
    quotas = np.floor(exact).astype(int)
    quotas[counts > 0] = np.maximum(quotas[counts > 0], 1)
    while int(quotas.sum()) > maximum:
        candidates = quotas[quotas > 1]
        quotas.loc[candidates.idxmax()] -= 1
    remainders = (exact - np.floor(exact)).sort_values(ascending=False)
    while int(quotas.sum()) < maximum:
        changed = False
        for label in remainders.index:
            if quotas.loc[label] < counts.loc[label]:
                quotas.loc[label] += 1
                changed = True
                if int(quotas.sum()) == maximum:
                    break
        if not changed:
            break
    parts = []
    for label, quota in quotas.items():
        group = frame[frame["family_label"] == label]
        parts.append(group.sample(n=int(quota), random_state=seed + int(label) * 1009))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_random_partitions(
    dataset_path: Path,
    feature_columns: list[str],
    caps: dict[str, int | None],
    seed: int,
) -> dict[str, pd.DataFrame]:
    columns = feature_columns + PROVENANCE_COLUMNS
    output = {}
    for split_name in ("train", "validation", "test"):
        frame = pd.read_parquet(
            dataset_path,
            columns=columns,
            filters=[("split_random", "==", split_name)],
        )
        output[split_name] = stratified_cap(frame, caps.get(split_name), seed)
    return output


def _read_days(
    dataset_path: Path, feature_columns: list[str], days: list[str]
) -> pd.DataFrame:
    columns = feature_columns + PROVENANCE_COLUMNS
    filters = [[("capture_day", "==", day)] for day in days]
    return pd.read_parquet(dataset_path, columns=columns, filters=filters)


def load_chronological_partitions(
    dataset_path: Path,
    feature_columns: list[str],
    caps: dict[str, int | None],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Compound day/campaign shift: Mon-Wed train, Thu validation, Fri test."""
    day_map = {
        "train": ["Monday", "Tuesday", "Wednesday"],
        "validation": ["Thursday"],
        "test": ["Friday"],
    }
    return {
        split: stratified_cap(
            _read_days(dataset_path, feature_columns, days), caps.get(split), seed
        )
        for split, days in day_map.items()
    }


def load_heldout_attack_partitions(
    dataset_path: Path,
    feature_columns: list[str],
    heldout_family: str,
    caps: dict[str, int | None],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Exclude one family from development; test only that family against benign."""
    columns = feature_columns + PROVENANCE_COLUMNS
    output = {}
    for split_name in ("train", "validation", "test"):
        frame = pd.read_parquet(
            dataset_path,
            columns=columns,
            filters=[("split_random", "==", split_name)],
        )
        if split_name == "test":
            frame = frame[frame["attack_class"].isin(["Benign", heldout_family])]
        else:
            frame = frame[frame["attack_class"] != heldout_family]
        output[split_name] = stratified_cap(frame, caps.get(split_name), seed)
    if output["test"]["binary_label"].nunique() != 2:
        raise ValueError(f"Held-out test for {heldout_family} does not contain both binary classes")
    return output


def git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def hardware_metadata() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def save_result(result: dict, output_dir: Path) -> Path:
    validate_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{result['experiment_id']}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    master_path = output_dir / "master_results.jsonl"
    records = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name.startswith("summary_"):
            continue
        candidate = json.loads(path.read_text(encoding="utf-8"))
        validate_result(candidate)
        records.append(candidate)
    master_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return result_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
