"""Cheap contract checks; independent of downloaded artifacts and large datasets."""
import io
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from src.demo.server import DemoService, make_handler, validate_frame


def test_feature_order_and_single_transform():
    frame = pd.DataFrame({"b": [-2.0] * 32, "a": [3.0] * 32})
    X = validate_frame(frame, ["a", "b"], 32)
    np.testing.assert_allclose(X[0], [np.log1p(3), -np.log1p(2)], rtol=1e-6)


@pytest.mark.parametrize("frame", [pd.DataFrame({"a": [1.0] * 31}),
    pd.DataFrame({"a": [np.nan] * 32}), pd.DataFrame({"a": [1.0] * 32, "label": [0] * 32}),
    pd.DataFrame({"a": ["bad"] * 32}), pd.DataFrame(np.ones((32, 2)), columns=["a", "a"])])
def test_invalid_feature_inputs(frame):
    with pytest.raises((ValueError, TypeError)):
        validate_frame(frame, ["a"], 32)


def test_unknown_source_rejected_before_loading(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        DemoService(tmp_path).load("../../outside")


def test_duplicate_upload_header_rejected():
    service = DemoService()
    service.load = lambda _: {"features": ["a"], "model": None}
    with pytest.raises(ValueError, match="Duplicate"):
        service.analyze({"csv": "a,a\n1,1\n"})


def test_evidence_reads_saved_ledger(tmp_path):
    (tmp_path / "results/metrics").mkdir(parents=True)
    (tmp_path / "results/tables").mkdir()
    (tmp_path / "results/figures").mkdir()
    ledger = tmp_path / "results/metrics/master_results.jsonl"
    ledger.write_text(json.dumps({"experiment_id": "example", "status": "complete"}) + "\n")
    before = ledger.read_bytes()
    result = DemoService(tmp_path).evidence()
    assert result["experimentCount"] == 1
    assert result["recoverySummary"] == {"runs": 0, "resumed": 0, "unsafe": 0}
    assert ledger.read_bytes() == before


def test_http_rejects_remote_origin(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DemoService(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert json.load(urlopen(base + "/api/health"))["status"] == "ok"
        req = Request(base + "/api/analyze", data=b"{}", headers={"Origin": "https://example.com", "Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(req)
        assert error.value.code == 403
        with pytest.raises(HTTPError):
            urlopen(base + "/api/figures/..%2Fprivate.png")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_incomplete_window_is_reported_not_classified():
    class Sentinel:
        window_size = 32

        def assess_window(self, X):
            assert len(X) == 32
            return {"status": "IN_DOMAIN", "shift_detected": False, "score": 0.01,
                    "threshold": 0.02, "rows": len(X)}

        def feature_scores(self, X):
            return np.array([0.01])

    class Guard:
        shift_sentinel = Sentinel()

        def predict_operational(self, X):
            assert len(X) == 32
            return np.full(len(X), "BENIGN")

    bundle = {"model": Guard(), "features": ["a"], "source": "source",
              "target": "target", "experimentId": "demo", "sha256": "test"}
    result = DemoService().infer(bundle, pd.DataFrame({"a": [1.0] * 33}), "unit check")
    assert result["scoredRows"] == 32
    assert result["pendingRows"] == 1
    assert result["counts"]["BENIGN"] == 32
    assert len(result["rows"]) == 32
