"""Presenter-paced recovery demonstration; no paper ledger/artifact writes."""
from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from experiments.domain_shift_common import label_blind_source_roles
from experiments.run_cross_netflow import signed_log1p
from experiments.run_domain_shift_validation import load_label_blind_partitions
from experiments.safe_resumption_common import allocate_recovery_roles, nested_balanced_review, choose_resumption_action, uncertainty_aware_gate_passes
from src.evaluation.abstention import fit_conformal_thresholds
from src.evaluation.research_metrics import binary_metrics
from src.models.abstention import AbstainingClassifier
from src.models.domain_shift import MarginalShiftSentinel, ShiftGuardedClassifier
from src.models.lightweight import build_lightweight_model
from src.models.recovery import ProbabilityRecalibrator

STAGES = ["Prepare disjoint data", "Train source and classify", "Detect changed domain",
          "Reveal review labels", "Train recovery candidates", "Validate independently",
          "Lock resumption decision", "Reveal untouched audit"]


def arrays(frame, features):
    return signed_log1p(frame[features].to_numpy(dtype=np.float64)), frame["binary_label"].to_numpy(dtype=np.int8)


def measure(model, X):
    """In-memory size and small warm single-row timing, not the paper benchmark."""
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=3)
    model.predict_proba(X[:1])
    timings = []
    for _ in range(40):
        start = time.perf_counter_ns()
        model.predict_proba(X[:1])
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return {"serialized_model_bytes": len(stream.getvalue()),
            "single_record_latency_ms_p95": float(np.percentile(timings, 95))}


class LifecycleDemo:
    """A single bounded in-memory session. Caller serializes access."""
    def __init__(self, root: Path):
        self.root = root
        self.session = None

    def start(self, source):
        if source not in ("nf_unsw_nb15_v2", "nf_ton_iot_v2"):
            raise ValueError("Unsupported source dataset")
        target = "nf_ton_iot_v2" if "unsw" in source else "nf_unsw_nb15_v2"
        self.session = dict(id=uuid.uuid4().hex, source=source, target=target, seed=15,
                            nextStage=0, events=[], status="ready", created=time.time())
        return self.snapshot()

    def snapshot(self):
        if self.session is None:
            raise ValueError("Start a demo session first")
        s = self.session
        return {key: s[key] for key in ("id", "source", "target", "seed", "nextStage", "events", "status")}

    def advance(self, session_id, expected_stage):
        s = self.session
        if s is None or s["id"] != session_id:
            raise ValueError("Session expired or replaced; start a new session")
        if s["status"] == "failed":
            raise ValueError("This session failed; restart rather than reuse partial state")
        stage = s["nextStage"]
        if expected_stage != stage or stage >= len(STAGES):
            raise ValueError("Stage mismatch; stages cannot be skipped or repeated")
        start = time.perf_counter()
        try:
            with threadpool_limits(limits=1):
                result = getattr(self, f"stage_{stage}")(s)
        except Exception:
            s["status"] = "failed"
            raise
        s["events"].append(dict(stage=stage, title=STAGES[stage], durationSeconds=time.perf_counter()-start, result=result))
        s["nextStage"] += 1
        s["status"] = "complete" if s["nextStage"] == len(STAGES) else "ready"
        return self.snapshot()

    def stage_0(self, s):
        config = yaml.safe_load((self.root / "configs/shadow_resumption.yaml").read_text())
        config["processed_dir"] = str(self.root / config["processed_dir"])
        metadata = json.loads((self.root / config["metadata_path"]).read_text())
        features = metadata["common_feature_columns"]
        parts, overlap = load_label_blind_partitions(config, s["source"], s["target"], s["seed"])
        settings = config["recovery"]
        roles = allocate_recovery_roles(parts["target"], settings["maximum_review_rows"],
            settings["validation_rows"], settings["final_audit_rows"], s["seed"], settings["shadow_validation_rows"])
        used = np.concatenate([r["recovery_original_index"].to_numpy() for r in roles.values()])
        if len(np.unique(used)) != len(used):
            raise ValueError("Recovery partitions overlap")
        # Monitor/replay rows are separate from review, both validation roles, and audit.
        unused = np.setdiff1d(np.arange(len(parts["target"])), used)
        window = config["domain_shift_validation"]["window_size"]
        monitor_index = np.random.default_rng(s["seed"] + 22001).permutation(unused)[:window]
        if len(monitor_index) != window:
            raise ValueError("Not enough independent monitoring rows")
        s.update(config=config, features=features, parts=parts, roles=roles,
                 monitor=parts["target"].iloc[monitor_index].copy())
        return dict(protocol="Live illustrative shadow-protocol run; label-blind canonical data; clean labels",
                    roleRows={**{k: len(v) for k,v in roles.items()}, "source_train":len(parts["train"]),
                              "source_validation":len(parts["validation"]), "monitor":window},
                    overlapRemoved=overlap, disjoint=True, audit="Sealed: not evaluated until after decision lock",
                    criteria=settings["gate"], labelBudgets=settings["label_budgets"])

    def stage_1(self, s):
        X, y = arrays(s["parts"]["train"], s["features"])
        V, vy = arrays(s["parts"]["validation"], s["features"])
        model = build_lightweight_model("decision_tree", s["seed"]).fit(X,y)
        settings = s["config"]["domain_shift_validation"]
        roles = label_blind_source_roles(len(V), settings["window_size"], s["seed"])
        sentinel = MarginalShiftSentinel(method="jsd_mean", window_size=settings["window_size"],
            quantile_bins=settings["quantile_bins"], calibration_alpha=settings["calibration_alpha"],
            maximum_reference_rows=settings["maximum_reference_rows"],
            minimum_calibration_windows=settings["minimum_calibration_windows"], seed=s["seed"])
        sentinel.fit(X).calibrate(V[roles["calibration"]], V[roles["audit"]])
        # Class rejection is calibrated on a third disjoint source role.
        rejection = roles["mixture"]
        thresholds = fit_conformal_thresholds(vy[rejection], model.predict_proba(V[rejection])[:,1],
                                              alpha=.10, class_conditional=True)
        guard = ShiftGuardedClassifier(AbstainingClassifier(model,"conformal",thresholds),sentinel)
        # Sentinel audit indices are relative to the separately supplied source audit role.
        window = V[roles["audit"]][sentinel.audit_window_indices_[0]]
        s.update(sourceModel=model, guard=guard)
        outputs = guard.predict_operational(window)
        return dict(model="Live-trained class-weighted decision tree", threshold=float(sentinel.threshold_),
                    assessment=sentinel.assess_window(window), counts=counts(outputs),
                    note="Source audit window 1, not cherry-picked; a legitimate source alarm is possible")

    def stage_2(self, s):
        X = signed_log1p(s["monitor"][s["features"]].to_numpy(dtype=np.float64))
        guard = s["guard"]
        assessment = guard.shift_sentinel.assess_window(X)
        s["shifted"] = assessment["shift_detected"]
        scores = guard.shift_sentinel.feature_scores(X)
        return dict(assessment=assessment, counts=counts(guard.predict_operational(X)),
                    contributors=sorted([dict(feature=f,score=float(v)) for f,v in zip(s["features"],scores)],
                                        key=lambda r:-r["score"])[:6],
                    note="Monitoring rows excluded from all recovery roles. A domain alarm is not an attack label.")

    def stage_3(self, s):
        s["reviewRevealed"] = True
        review = s["roles"]["review"]
        return dict(rows=len(review), classCounts={str(k):int(v) for k,v in review["binary_label"].value_counts().items()},
                    note="Labels revealed from dataset ground truth for rehearsal; no live human-labeling claim.")

    def stage_4(self, s):
        if not s.get("shifted"):
            s["candidates"] = []
            return dict(candidates=[], note="No shift detected; recovery skipped rather than forced.")
        candidates = [dict(action="no_adaptation", label_budget=0, model=s["sourceModel"], trainingSeconds=0)]
        for budget in s["config"]["recovery"]["label_budgets"]:
            review = nested_balanced_review(s["roles"]["review"],budget,s["seed"])
            X,y = arrays(review,s["features"])
            for action in ("recalibration","compact_retraining"):
                started=time.perf_counter()
                model = ProbabilityRecalibrator(s["sourceModel"],s["seed"]+budget) if action=="recalibration" else build_lightweight_model("decision_tree",s["seed"]+budget)
                model.fit(X,y)
                candidates.append(dict(action=action,label_budget=budget,model=model,trainingSeconds=time.perf_counter()-started))
        s["candidates"] = candidates
        return dict(candidates=[{k:c[k] for k in ("action","label_budget","trainingSeconds")} for c in candidates],
                    note="Fresh in-memory fitting; no saved experiment/artifact overwritten.")

    def stage_5(self, s):
        X,y = arrays(s["roles"]["validation"],s["features"])
        Z,zy = arrays(s["roles"]["shadow_validation"],s["features"])
        criteria = s["config"]["recovery"]["gate"]
        actions=[]
        for c in s["candidates"]:
            model=c["model"]
            efficiency=measure(model,X)
            vm=binary_metrics(y,model.predict_proba(X)[:,1])
            sm=binary_metrics(zy,model.predict_proba(Z)[:,1])
            vp,vu=uncertainty_aware_gate_passes(vm,efficiency,criteria)
            sp,su=uncertainty_aware_gate_passes(sm,efficiency,criteria)
            actions.append(dict(action=c["action"],label_budget=c["label_budget"],
                validation_metrics=vm,shadow_metrics=sm,efficiency=efficiency,
                validation_passed=vp,shadow_passed=sp,gate_passed=bool(vp and sp),
                validationFprUpper=vu,shadowFprUpper=su))
        s["actions"]=actions
        return dict(actions=actions, criteria=criteria, audit="Still sealed",
                    measurement="Compressed in-memory joblib bytes; 40 warm single-row timings on selection validation, one CPU thread. Demo check, not paper benchmark.")

    def stage_6(self, s):
        selected=choose_resumption_action(s["actions"])
        s["selected"]=selected
        s["decisionLocked"]=True
        if selected is None:
            return dict(decision="RECALIBRATION_REQUIRED" if s["shifted"] else "NO_RECOVERY_NEEDED",
                        locked=True, note="Source deployment artifact unchanged. No audit-driven retries.")
        model=next(c["model"] for c in s["candidates"] if c["action"]==selected["action"] and c["label_budget"]==selected["label_budget"])
        s["selectedModel"]=model
        X=signed_log1p(s["monitor"][s["features"]].to_numpy(dtype=np.float64))
        decisions=np.where(model.predict_proba(X)[:,1]>=.5,"MALICIOUS","BENIGN")
        return dict(decision="SAFE_TO_RESUME",action=selected["action"],labels=selected["label_budget"],
                    locked=True,counts=counts(decisions),
                    note="Conditional target-model classification inside this session only. Default binary threshold; no new conformal/target sentinel guarantee or deployment activation.")

    def stage_7(self, s):
        if not s.get("decisionLocked"):
            raise ValueError("Audit cannot be opened before decision lock")
        X,y=arrays(s["roles"]["audit"],s["features"])
        baseline=binary_metrics(y,s["sourceModel"].predict_proba(X)[:,1])
        selected=s["selected"]
        if selected is None:
            return dict(baseline=baseline,selected=None,unsafeResumption=False, note="No candidate resumed. Audit cannot alter the locked decision.")
        metrics=binary_metrics(y,s["selectedModel"].predict_proba(X)[:,1])
        passed,upper=uncertainty_aware_gate_passes(metrics,selected["efficiency"],s["config"]["recovery"]["gate"])
        return dict(baseline=baseline,selected=metrics,auditPassed=passed,auditFprUpper=upper,
                    unsafeResumption=not passed,note="Reporting-only audit. Failure retained; no reselection. This single demonstration is not new paper evidence.")


def counts(outputs):
    return {state:int((outputs==state).sum()) for state in ("BENIGN","MALICIOUS","UNKNOWN","DOMAIN_SHIFT_DETECTED")}
