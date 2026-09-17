import json
from pathlib import Path

import pytest

from experiments import research_common as common


def record():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "results/metrics/master_results.jsonl").read_text(
        encoding="utf-8").splitlines()[0])


def test_failed_atomic_replace_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    destination = tmp_path / "ledger.jsonl"
    destination.write_text("original", encoding="utf-8")
    def fail(*args):
        raise OSError("simulated interrupted replacement")
    monkeypatch.setattr(common.os, "replace", fail)
    with pytest.raises(OSError):
        common.atomic_write_text(destination, "replacement")
    assert destination.read_text() == "original"
    assert list(tmp_path.iterdir()) == [destination]


def test_rebuild_recovers_missing_ledger_without_retraining(tmp_path):
    item = record()
    (tmp_path / f"{item['experiment_id']}.json").write_text(json.dumps(item))
    assert common.rebuild_results_ledger(tmp_path) == 1
    before = (tmp_path / "master_results.jsonl").read_bytes()
    assert common.rebuild_results_ledger(tmp_path) == 1
    assert before == (tmp_path / "master_results.jsonl").read_bytes()


def test_invalid_record_cannot_destroy_previous_ledger(tmp_path):
    ledger = tmp_path / "master_results.jsonl"
    ledger.write_text("previous index")
    (tmp_path / "broken.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        common.rebuild_results_ledger(tmp_path)
    assert ledger.read_text() == "previous index"


def test_result_id_cannot_escape_output_directory(tmp_path):
    item = record()
    item["experiment_id"] = "../outside"
    with pytest.raises(ValueError, match="Experiment ID"):
        common.save_result(item, tmp_path)
    assert not list(tmp_path.iterdir())
