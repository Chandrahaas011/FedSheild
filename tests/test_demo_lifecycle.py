"""Lifecycle order and audit-isolation checks, without large local datasets."""
import numpy as np
import pytest

import src.demo.lifecycle as lifecycle


def test_stages_cannot_skip_repeat_or_cross_sessions(tmp_path, monkeypatch):
    demo = lifecycle.LifecycleDemo(tmp_path)
    first = demo.start("nf_unsw_nb15_v2")
    with pytest.raises(ValueError, match="Stage mismatch"):
        demo.advance(first["id"], 1)
    monkeypatch.setattr(demo, "stage_0", lambda s: {"audit": "sealed"})
    result = demo.advance(first["id"], 0)
    assert result["nextStage"] == 1
    assert "parts" not in result
    with pytest.raises(ValueError, match="Stage mismatch"):
        demo.advance(first["id"], 0)
    demo.start("nf_ton_iot_v2")
    with pytest.raises(ValueError, match="expired"):
        demo.advance(first["id"], 1)


def test_audit_requires_decision_lock(tmp_path):
    with pytest.raises(ValueError, match="before decision lock"):
        lifecycle.LifecycleDemo(tmp_path).stage_7({})


def test_validation_never_reads_audit(tmp_path, monkeypatch):
    audit = object()
    validation, shadow = object(), object()
    seen = []

    def array_spy(frame, features):
        assert frame is not audit
        seen.append(frame)
        return np.zeros((40, 2)), np.tile([0, 1], 20)

    class Model:
        def predict_proba(self, X):
            p = np.tile([.05, .95], len(X) // 2)
            return np.column_stack([1-p, p])

    monkeypatch.setattr(lifecycle, "arrays", array_spy)
    monkeypatch.setattr(lifecycle, "measure", lambda *args: {"serialized_model_bytes": 100, "single_record_latency_ms_p95": .1})
    criteria = dict(minimum_macro_f1=.8, minimum_malicious_recall=.8,
        maximum_false_positive_rate=.2, maximum_ece=.1, maximum_model_bytes=1024,
        maximum_single_record_latency_p95_ms=5)
    state = dict(roles={"validation": validation, "shadow_validation": shadow, "audit": audit},
        features=["a", "b"], config={"recovery": {"gate": criteria}},
        candidates=[dict(model=Model(),action="compact_retraining",label_budget=100)])
    result = lifecycle.LifecycleDemo(tmp_path).stage_5(state)
    assert seen == [validation, shadow]
    assert result["audit"] == "Still sealed"
    assert "final_audit_metrics" not in result["actions"][0]


def test_failed_stage_cannot_retry_partial_state(tmp_path, monkeypatch):
    demo = lifecycle.LifecycleDemo(tmp_path)
    session = demo.start("nf_unsw_nb15_v2")
    def fail(state):
        raise ValueError("failure")
    monkeypatch.setattr(demo, "stage_0", fail)
    with pytest.raises(ValueError, match="failure"):
        demo.advance(session["id"], 0)
    assert demo.snapshot()["status"] == "failed"
    with pytest.raises(ValueError, match="restart"):
        demo.advance(session["id"], 0)


def test_selection_ignores_audit_fields(tmp_path):
    criteria = {"serialized_model_bytes": 100}
    actions = [dict(action="first",label_budget=100,gate_passed=True,
                    efficiency=criteria,validation_metrics={"macro_f1": .9},
                    final_audit_metrics={"macro_f1": 0}),
               dict(action="second",label_budget=500,gate_passed=True,
                    efficiency=criteria,validation_metrics={"macro_f1": 1},
                    final_audit_metrics={"macro_f1": 1})]
    assert lifecycle.choose_resumption_action(actions)["action"] == "first"


def test_unknown_source_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        lifecycle.LifecycleDemo(tmp_path).start("unknown")
