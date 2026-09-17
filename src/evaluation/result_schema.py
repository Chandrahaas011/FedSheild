from __future__ import annotations


REQUIRED_TOP_LEVEL = {
    "experiment_id",
    "created_at",
    "status",
    "code_version",
    "dataset",
    "protocol",
    "seed",
    "model",
    "feature_set",
    "metrics",
    "efficiency",
    "durations",
    "hardware",
}


def validate_result(result: dict) -> None:
    missing = REQUIRED_TOP_LEVEL - set(result)
    if missing:
        raise ValueError(f"Result is missing required fields: {sorted(missing)}")
    if result["status"] not in {"complete", "failed"}:
        raise ValueError("Result status must be 'complete' or 'failed'")
    if result["status"] == "complete":
        for key in ("macro_f1", "mcc", "false_positive_rate", "brier_score"):
            if key not in result["metrics"]:
                raise ValueError(f"Completed result is missing metric: {key}")
        if result["feature_set"]["count"] <= 0:
            raise ValueError("Feature count must be positive")
        if result["experiment_id"].startswith("domain_shift_"):
            if "domain_shift" not in result:
                raise ValueError("Domain-shift result is missing domain_shift metrics")
            shift = result["domain_shift"]
            for role in (
                "source_audit",
                "mixed_target",
                "benign_target_diagnostic",
                "malicious_target_diagnostic",
            ):
                metrics = shift.get("primary", {}).get(role, {})
                rate = metrics.get("alarm_rate")
                if rate is None or not 0.0 <= rate <= 1.0:
                    raise ValueError(f"Invalid domain-shift alarm rate for {role}")
                if metrics.get("windows", 0) <= 0:
                    raise ValueError(f"Domain-shift role has no windows: {role}")
            output_rates = shift.get("guarded_operational_outputs", {}).get(
                "output_rates", {}
            )
            required_states = {
                "BENIGN", "MALICIOUS", "UNKNOWN", "DOMAIN_SHIFT_DETECTED"
            }
            if set(output_rates) != required_states:
                raise ValueError("Guarded outputs must contain exactly four states")
            if abs(sum(output_rates.values()) - 1.0) > 1.0e-9:
                raise ValueError("Guarded output rates must sum to one")
        if result["experiment_id"].startswith("domain_validation_"):
            validation = result.get("domain_shift_validation", {})
            if result["experiment_id"].startswith(
                "domain_validation_window_sensitivity_"
            ) and validation.get("window_size") not in {128, 256, 512, 1024}:
                raise ValueError("Window-sensitivity record has an unfrozen window size")
            methods = validation.get("methods", {})
            if set(methods) != {"jsd_mean", "psi_mean", "ks_mean"}:
                raise ValueError("Domain-validation record must compare JSD, PSI, and KS")
            for name, method in methods.items():
                for role in ("source_audit", "target_full"):
                    rate = method.get(role, {}).get("alarm_rate")
                    if rate is None or not 0.0 <= rate <= 1.0:
                        raise ValueError(f"Invalid {name} alarm rate for {role}")
                if "0.5" not in method.get("mixtures", {}):
                    raise ValueError(f"{name} is missing the 50 percent target mixture")
        if result["experiment_id"].startswith((
            "safe_resumption_", "prospective_resumption_", "shadow_resumption_",
            "label_noise_resumption_", "robust_noise_resumption_",
            "asymmetric_noise_resumption_"
        )):
            recovery = result.get("safe_resumption", {})
            actions = recovery.get("actions", [])
            expected = {
                ("no_adaptation", 0),
                ("recalibration", 100),
                ("recalibration", 500),
                ("recalibration", 2000),
                ("compact_retraining", 100),
                ("compact_retraining", 500),
                ("compact_retraining", 2000),
            }
            if result["experiment_id"].startswith((
                "robust_noise_resumption_", "asymmetric_noise_resumption_"
            )):
                expected |= {
                    ("robust_compact_retraining", 100),
                    ("robust_compact_retraining", 500),
                    ("robust_compact_retraining", 2000),
                }
            observed = {(action.get("action"), action.get("label_budget")) for action in actions}
            if observed != expected:
                raise ValueError("Safe-resumption record has an incomplete action/budget matrix")
            for action in actions:
                for role in ("validation_metrics", "final_audit_metrics"):
                    metrics = action.get(role, {})
                    for key in ("macro_f1", "false_positive_rate", "ece_15"):
                        value = metrics.get(key)
                        if value is None or not 0.0 <= value <= 1.0:
                            raise ValueError(f"Safe-resumption action has invalid {role} {key}")
                if action.get("unsafe_resumption") != bool(
                    action.get("gate_passed") and not action.get("final_audit_passed")
                ):
                    raise ValueError("Unsafe-resumption flag is inconsistent with gate outcomes")
            selected = recovery.get("selected_policy", {})
            if selected.get("decision") not in {"SAFE_TO_RESUME", "RECALIBRATION_REQUIRED"}:
                raise ValueError("Safe-resumption decision is invalid")
            if selected.get("decision") == "SAFE_TO_RESUME" and not selected.get("gate_passed"):
                raise ValueError("SAFE_TO_RESUME requires a passing validation gate")
            if result["experiment_id"].startswith("prospective_resumption_"):
                if recovery.get("gate_mode") != "wilson_fpr_upper_95":
                    raise ValueError("Prospective resumption requires the frozen Wilson FPR gate")
                for action in actions:
                    if action.get("validation_fpr_wilson_upper_95") is None:
                        raise ValueError("Prospective action is missing its validation FPR bound")
            if result["experiment_id"].startswith("shadow_resumption_"):
                if recovery.get("gate_mode") != "wilson_fpr_upper_95":
                    raise ValueError("Shadow resumption requires the frozen Wilson FPR gate")
                for action in actions:
                    if action.get("shadow_validation_metrics") is None:
                        raise ValueError("Shadow action is missing independent validation metrics")
                    if action.get("shadow_validation_passed") is None:
                        raise ValueError("Shadow action is missing its second gate decision")
            if result["experiment_id"].startswith("label_noise_resumption_"):
                rate = recovery.get("review_label_noise_rate")
                if rate not in {0.0, 0.05, 0.1, 0.2}:
                    raise ValueError("Label-noise resumption has an unfrozen noise rate")
                for action in actions:
                    if action.get("shadow_validation_metrics") is None:
                        raise ValueError("Label-noise action is missing shadow validation")
            if result["experiment_id"].startswith("robust_noise_resumption_"):
                rate = recovery.get("review_label_noise_rate")
                if rate not in {0.0, 0.05, 0.1, 0.2}:
                    raise ValueError("Robust-noise resumption has an unfrozen noise rate")
                for action in actions:
                    if action.get("shadow_validation_metrics") is None:
                        raise ValueError("Robust-noise action is missing shadow validation")
            if result["experiment_id"].startswith("asymmetric_noise_resumption_"):
                rates = recovery.get("review_label_noise_by_class")
                allowed = (
                    {"0": 0.05, "1": 0.0}, {"0": 0.1, "1": 0.0},
                    {"0": 0.0, "1": 0.05}, {"0": 0.0, "1": 0.1},
                    {0: 0.05, 1: 0.0}, {0: 0.1, 1: 0.0},
                    {0: 0.0, 1: 0.05}, {0: 0.0, 1: 0.1},
                )
                if rates not in allowed:
                    raise ValueError("Asymmetric-noise record has an unfrozen profile")
                for action in actions:
                    if action.get("shadow_validation_metrics") is None:
                        raise ValueError("Asymmetric-noise action is missing shadow validation")
