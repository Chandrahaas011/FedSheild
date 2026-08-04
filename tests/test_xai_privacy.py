import numpy as np

import src.xai.federated_shap as federated_shap


def test_federated_xai_default_returns_summaries_without_raw_examples(monkeypatch):
    """The server-facing default must never collect client-level SHAP records."""
    calls = []

    def fake_explain(*args, collect_local_examples=False, **kwargs):
        calls.append(collect_local_examples)
        return {
            "summary": {"Attack": {"mean_abs_shap": [1.0, 2.0], "n_samples": 1}},
            "local_examples": [{"feature_values": [99.0, 42.0], "shap_values": [3.0, 4.0]}],
        }

    monkeypatch.setattr(federated_shap, "explain_client", fake_explain)
    summary, local_examples = federated_shap.run_federated_shap(
        model=None,
        X=np.zeros((4, 2), dtype=np.float32),
        client_indices=[np.array([0, 1]), np.array([2, 3])],
        feature_names=["a", "b"],
        class_names=["Benign", "Attack"],
    )

    assert calls == [False, False]
    assert local_examples == []
    assert summary["Attack"]["n_samples"] == 2


def test_federated_xai_debug_export_is_explicit(monkeypatch):
    def fake_explain(*args, collect_local_examples=False, **kwargs):
        return {
            "summary": {"Attack": {"mean_abs_shap": [1.0], "n_samples": 1}},
            "local_examples": [{"feature_values": [7.0], "shap_values": [2.0]}] if collect_local_examples else [],
        }

    monkeypatch.setattr(federated_shap, "explain_client", fake_explain)
    _, local_examples = federated_shap.run_federated_shap(
        model=None,
        X=np.zeros((2, 1), dtype=np.float32),
        client_indices=[np.array([0, 1])],
        feature_names=["a"],
        class_names=["Benign", "Attack"],
        collect_local_examples=True,
    )

    assert local_examples == [{"feature_values": [7.0], "shap_values": [2.0]}]
