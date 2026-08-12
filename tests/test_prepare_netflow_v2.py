from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.data.prepare_netflow_v2 import canonicalize_archive


def test_netflow_builder_excludes_identifiers_and_removes_candidate_contradictions(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["1.1.1.1"] * 5,
            "L4_SRC_PORT": [10, 10, 11, 12, 13],
            "F1": [1, 1, 2, 3, 4],
            "F2": [5, 5, 6, 7, 8],
            "Label": [0, 1, 0, 1, 1],
            "Attack": ["Benign", "Attack", "Benign", "Attack", "Attack"],
        }
    )
    csv_path = tmp_path / "flows.csv"
    frame.to_csv(csv_path, index=False)
    archive_path = tmp_path / "flows.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, "data/flows.csv")
    output_path = tmp_path / "canonical.parquet"
    metadata = canonicalize_archive(
        archive_path,
        "tiny",
        output_path,
        ["IPV4_SRC_ADDR", "L4_SRC_PORT"],
        maximum_rows_per_class=100,
        chunk_rows=2,
    )
    canonical = pd.read_parquet(output_path)
    assert metadata["candidate_contradictory_fingerprints"] == 1
    assert metadata["canonical_rows"] == 3
    assert metadata["feature_columns"] == ["F1", "F2"]
    assert "IPV4_SRC_ADDR" not in canonical.columns
