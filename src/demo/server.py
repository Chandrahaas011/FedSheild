"""Loopback-only API: saved results and isolated in-memory recovery demonstrations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from experiments.run_cross_netflow import load_cross_partitions, signed_log1p
from src.demo.lifecycle import LifecycleDemo

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Know When to Resume: Validation-Gated, Resource-Bounded Recovery for Lightweight IoT Intrusion Detection"
STATES = ("BENIGN", "MALICIOUS", "UNKNOWN", "DOMAIN_SHIFT_DETECTED")
MAX_BODY = 4 * 1024 * 1024
MAX_ROWS = 8192
SOURCES = ("nf_unsw_nb15_v2", "nf_ton_iot_v2")


def validate_frame(frame: pd.DataFrame, features: list[str], window: int) -> np.ndarray:
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed")
    if set(frame.columns) != set(features):
        missing = sorted(set(features) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(features))
        raise ValueError(f"Expected only the {len(features)} raw feature columns. Missing: {missing}; extra: {extra}")
    if not window <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Provide between {window} and {MAX_ROWS} rows")
    raw = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Features must be numeric and finite; missing values are not accepted")
    return signed_log1p(raw)


class DemoService:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.cache = {}
        self.lock = threading.RLock()
        self.lifecycle = LifecycleDemo(root)

    def load(self, source: str):
        if source not in SOURCES:
            raise ValueError("Unsupported source dataset")
        if source in self.cache:
            return self.cache[source]
        target = next(s for s in SOURCES if s != source)
        experiment_id = f"domain_shift_{source}_to_{target}_hgb_seed0"
        record = json.loads((self.root / "results/metrics" / f"{experiment_id}.json").read_text())
        path = self.root / "models/artifacts" / f"{experiment_id}.joblib"
        # Only a fixed repository artifact can be deserialized, never an uploaded pickle.
        model = joblib.load(path)
        features = record["feature_set"]["names"]
        metadata = json.loads((self.root / "data/processed/netflow_v2/metadata.json").read_text())
        if features != metadata["common_feature_columns"]:
            raise ValueError("Processed-data feature order differs from the saved experiment")
        if model.shift_sentinel.n_features_in_ != len(features):
            raise ValueError("Artifact feature count does not match experiment metadata")
        if model.shift_sentinel.window_size != record["efficiency"]["window_size"]:
            raise ValueError("Artifact window size does not match experiment metadata")
        bundle = dict(model=model, features=features, record=record, source=source,
                      target=target, experimentId=experiment_id,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.cache[source] = bundle
        return bundle

    def schema(self, source: str):
        with self.lock:
            b = self.load(source)
            sentinel = b["model"].shift_sentinel
            return dict(source=source, target=b["target"], features=b["features"],
                        windowSize=sentinel.window_size, threshold=float(sentinel.threshold_),
                        experimentId=b["experimentId"], artifactSha256=b["sha256"],
                        modelBytes=b["record"]["efficiency"]["serialized_model_bytes"],
                        protocol="Historical class-capped NetFlow-v2; source-only 512-flow guard; seed 0")

    def evidence(self):
        ledger = self.root / "results/metrics/master_results.jsonl"
        records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        figures = sorted(p.name for p in (self.root / "results/figures").glob("research_figure*.png"))
        table = self.root / "results/tables/research_shadow_resumption_confirmation.csv"
        recovery = list(csv.DictReader(io.StringIO(table.read_text()))) if table.exists() else []
        resumptions = [r for r in recovery if r["shadow_decision"] == "SAFE_TO_RESUME"]
        return dict(title=TITLE, experimentCount=len(records), figureCount=len(figures),
                    tableCount=len(list((self.root / "results/tables").glob("research_*.csv"))),
                    figures=figures, recovery=recovery,
                    recoverySummary=dict(runs=len(recovery), resumed=len(resumptions),
                        unsafe=sum(int(r["shadow_unsafe_resumption"]) for r in recovery)),
                    experiments=[dict(id=r["experiment_id"], status=r.get("status", "unknown"),
                        seed=r.get("seed"), model=r.get("model", {}).get("name", "—"),
                        protocol=r.get("protocol", {}).get("name", "—")) for r in records])

    def replay_frame(self, bundle, scenario: str, offset: int):
        if scenario not in ("source", "target", "mixture"):
            raise ValueError("Scenario must be source, target, or mixture")
        if not 0 <= offset < 20:
            raise ValueError("Window index must be between 0 and 19")
        if "parts" not in bundle:
            config = yaml.safe_load((self.root / "configs/netflow_v2_cross.yaml").read_text())
            config["processed_dir"] = str(self.root / config["processed_dir"])
            config["target_cap_strategy"] = "label_blind_random"
            parts, _ = load_cross_partitions(config, bundle["source"], bundle["target"], 0)
            bundle["parts"] = parts
        parts = bundle["parts"]
        sentinel = bundle["model"].shift_sentinel
        # Sequential audit windows, not a cherry-picked passing example.
        source_index = sentinel.audit_window_indices_[offset % len(sentinel.audit_window_indices_)]
        source_frame = parts["validation"].iloc[source_index]
        observed = sentinel.score_window(signed_log1p(source_frame[bundle["features"]].to_numpy(dtype=np.float64)))
        expected = sentinel.audit_scores_[offset % len(sentinel.audit_scores_)]
        if not np.isclose(observed, expected, atol=1e-8, rtol=1e-6):
            raise ValueError("Recreated source audit window does not match the saved artifact; check data/config provenance")
        # Same label-blind permutation used for saved mixed-target evaluation.
        indices = np.random.default_rng(8101).permutation(len(parts["test"]))
        n = sentinel.window_size
        target_frame = parts["test"].iloc[indices[offset * n:(offset + 1) * n]]
        if scenario == "source":
            return source_frame
        if scenario == "target":
            return target_frame
        mixture = pd.concat([source_frame.iloc[:n // 2], target_frame.iloc[:n // 2]])
        return mixture.sample(frac=1, random_state=offset + 9101)

    def infer(self, bundle, frame: pd.DataFrame, origin: str):
        model = bundle["model"]
        window = model.shift_sentinel.window_size
        X = validate_frame(frame[bundle["features"]], bundle["features"], window)
        windows = []
        counts = dict.fromkeys(STATES, 0)
        rows = []
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            for start in range(0, len(X) - window + 1, window):
                chunk = X[start:start + window]
                assessment = model.shift_sentinel.assess_window(chunk)
                outputs = model.predict_operational(chunk)
                feature_scores = model.shift_sentinel.feature_scores(chunk)
                top = sorted(zip(bundle["features"], feature_scores), key=lambda x: -x[1])[:6]
                windows.append(dict(index=len(windows) + 1, **assessment,
                                    contributors=[dict(feature=f, score=float(s)) for f, s in top]))
                for state in STATES:
                    counts[state] += int((outputs == state).sum())
                if len(rows) < 80:
                    rows.extend(dict(row=start + i + 1, decision=str(value))
                                for i, value in enumerate(outputs[:80 - len(rows)]))
        elapsed = (time.perf_counter() - started) * 1000
        return dict(origin=origin, source=bundle["source"], target=bundle["target"],
                    experimentId=bundle["experimentId"], artifactSha256=bundle["sha256"],
                    inputRows=len(X), scoredRows=sum(counts.values()),
                    pendingRows=len(X) % window, counts=counts, windows=windows, rows=rows,
                    elapsedMs=elapsed, timingScope="Server analysis including repeated diagnostic scoring; not a research latency benchmark",
                    note="Replay is an illustration of a saved protocol, not a new experimental result. Labels are never inference inputs.")

    def analyze(self, payload):
        with self.lock:
            bundle = self.load(payload.get("source", SOURCES[0]))
            if "csv" in payload:
                text = payload["csv"]
                if not isinstance(text, str):
                    raise ValueError("CSV must be text")
                header = next(csv.reader(io.StringIO(text)), [])
                if len(header) != len(set(header)):
                    raise ValueError("Duplicate CSV headers are not allowed")
                frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1)
                validate_frame(frame, bundle["features"], bundle["model"].shift_sentinel.window_size)
                return self.infer(bundle, frame, "Uploaded raw-feature CSV")
            scenario = payload.get("scenario", "source")
            offset = int(payload.get("offset", 0))
            frame = self.replay_frame(bundle, scenario, offset)
            return self.infer(bundle, frame, f"{scenario} replay · window {offset + 1}")


def make_handler(service: DemoService):
    class Handler(BaseHTTPRequestHandler):
        def local_host(self):
            host = self.headers.get("Host", "")
            return host in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def respond(self, status, body, content_type="application/json"):
            data = json.dumps(body, allow_nan=False).encode() if content_type == "application/json" else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.local_host():
                return self.respond(403, dict(error="Local Host header required"))
            route = urlparse(self.path).path
            try:
                if route == "/api/health":
                    return self.respond(200, dict(status="ok", mode="local demo; research evidence unchanged"))
                if route == "/api/evidence":
                    return self.respond(200, service.evidence())
                if route == "/api/schema":
                    return self.respond(200, service.schema(SOURCES[0]))
                if route.startswith("/api/sample/"):
                    source = route.rsplit("/", 1)[-1].removesuffix(".csv")
                    with service.lock:
                        bundle = service.load(source)
                        frame = service.replay_frame(bundle, "source", 0)
                        sample = frame[bundle["features"]].to_csv(index=False).encode()
                    return self.respond(200, sample, "text/csv; charset=utf-8")
                if route.startswith("/api/") and not route.startswith("/api/figures/"):
                    return self.respond(404, dict(error="Unknown endpoint"))
                if route.startswith("/api/figures/"):
                    name = route.rsplit("/", 1)[-1]
                    if not re.fullmatch(r"research_figure[0-9]+_[a-zA-Z0-9_]+\.png", name):
                        raise ValueError("Invalid figure name")
                    path = service.root / "results/figures" / name
                else:
                    dist = service.root / "ui/dist"
                    path = (dist / route.lstrip("/")).resolve()
                    if not path.is_relative_to(dist.resolve()):
                        raise ValueError("Invalid path")
                    if not path.is_file():
                        path = dist / "index.html"
                if not path.is_file():
                    return self.respond(404, dict(error="File not available; build the UI first"))
                return self.respond(200, path.read_bytes(), mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            except (ValueError, FileNotFoundError) as exc:
                self.respond(400, dict(error=str(exc)))

        def do_POST(self):
            if not self.local_host():
                return self.respond(403, dict(error="Local Host header required"))
            # No remote origins; Vite proxies same-origin requests during development.
            origin = self.headers.get("Origin")
            port = self.server.server_port
            if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}", "http://127.0.0.1:5173", "http://localhost:5173"):
                return self.respond(403, dict(error="Only local demonstration origins are allowed"))
            if self.path not in ("/api/analyze", "/api/lifecycle/start", "/api/lifecycle/next"):
                return self.respond(404, dict(error="Unknown endpoint"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, dict(error="Request must be between 1 byte and 4 MiB"))
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Use application/json")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/analyze":
                    result = service.analyze(payload)
                else:
                    with service.lock:
                        if self.path.endswith("/start"):
                            result = service.lifecycle.start(payload.get("source", SOURCES[0]))
                        else:
                            result = service.lifecycle.advance(payload.get("id"), payload.get("expectedStage"))
                self.respond(200, result)
            except (ValueError, TypeError, KeyError, FileNotFoundError, UnicodeError) as exc:
                self.respond(400, dict(error=str(exc)))
            except Exception:
                self.log_error("Analysis failed; check local artifact compatibility")
                self.respond(500, dict(error="Analysis failed. Check the terminal and local model/data setup."))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(DemoService()))
    print(f"Know When to Resume · http://127.0.0.1:{args.port} · Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
