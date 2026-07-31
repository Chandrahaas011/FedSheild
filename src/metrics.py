"""Evaluation metrics shared by centralized training, FL simulation, and table generation.

Accuracy is intentionally secondary: ~80% of CICIDS2017 is BENIGN, so a majority-class
predictor scores ~80% accuracy while missing every attack. Macro-F1, per-class recall,
and FPR (false-alarm rate, the SOC-analyst-relevant number) are the metrics that matter.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], benign_idx: int = 0) -> dict:
    n_classes = len(class_names)
    labels = list(range(n_classes))

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean())

    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_fpr = {}
    for i in range(n_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        per_class_fpr[class_names[i]] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    y_true_bin = (y_true != benign_idx).astype(int)
    y_pred_bin = (y_pred != benign_idx).astype(int)
    fp_benign = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    tn_benign = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    overall_fpr = float(fp_benign / (fp_benign + tn_benign)) if (fp_benign + tn_benign) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "per_class_recall": {class_names[i]: float(per_class_recall[i]) for i in range(n_classes)},
        "per_class_fpr": per_class_fpr,
        "overall_fpr": overall_fpr,
        "confusion_matrix": cm.tolist(),
    }


def summarize_over_seeds(metric_dicts: list[dict]) -> dict:
    """Mean +/- std across a list of compute_metrics() outputs (one per seed)."""
    scalar_keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "overall_fpr"]
    summary = {}
    for k in scalar_keys:
        vals = np.array([m[k] for m in metric_dicts])
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std())

    class_names = list(metric_dicts[0]["per_class_recall"].keys())
    summary["per_class_recall_mean"] = {
        c: float(np.mean([m["per_class_recall"][c] for m in metric_dicts])) for c in class_names
    }
    summary["per_class_recall_std"] = {
        c: float(np.std([m["per_class_recall"][c] for m in metric_dicts])) for c in class_names
    }
    summary["per_class_fpr_mean"] = {
        c: float(np.mean([m["per_class_fpr"][c] for m in metric_dicts])) for c in class_names
    }
    return summary
