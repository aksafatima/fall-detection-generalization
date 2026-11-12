"""
Evaluation metrics for fall detection.

Primary metric: sensitivity (recall) — missing a fall is costly.
Also reports: specificity, precision, F1, balanced accuracy, MCC, PR-AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Compute all thesis evaluation metrics.

    Parameters
    ----------
    y_true : (N,) ground truth binary labels
    y_pred : (N,) predicted binary labels
    y_prob : (N,) predicted probability of fall class (optional)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "false_alarm_rate": false_alarm_rate,
        "precision":        precision_score(y_true, y_pred, zero_division=0),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "balanced_acc":     balanced_accuracy_score(y_true, y_pred),
        "mcc":              matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    if y_prob is not None:
        metrics["pr_auc"]  = average_precision_score(y_true, y_prob)
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval for a scalar metric.

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            score = np.nan
        boot_scores.append(score)

    boot_scores = np.array(boot_scores)
    point = metric_fn(y_true, y_pred)
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(boot_scores, [100 * alpha, 100 * (1 - alpha)])
    return float(point), float(lo), float(hi)


def metrics_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert a list of per-fold metric dicts to a tidy DataFrame."""
    return pd.DataFrame(results)
