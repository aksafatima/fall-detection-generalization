"""
Evaluation protocols P1, P2, P3 as defined in Delivery 1 §5.

P1 — Within-dataset, subject-independent (grouped k-fold CV)
P2 — Cross-population (train young, test elderly; and reverse)
P3 — Cross-dataset (train on dataset A, test on dataset B)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

from .metrics import compute_metrics


def _fit_predict(model, X_tr, y_tr, X_te):
    """Fit model and return (y_pred, y_prob)."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = None
    return y_pred, y_prob


def run_p1_within(
    model,
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    n_splits: int = 5,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    P1: Within-dataset subject-independent evaluation.

    GroupKFold ensures no subject appears in both train and test within a fold.
    Windows of the same subject are treated as a group.

    Returns
    -------
    DataFrame with one row per fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = np.array(subjects)

    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        m = clone(model)
        y_pred, y_prob = _fit_predict(m, X_tr, y_tr, X_te)
        metrics = compute_metrics(y_te, y_pred, y_prob)
        metrics.update({"protocol": "P1", "fold": fold, "model": model_name,
                        "n_train": len(tr_idx), "n_test": len(te_idx)})
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)


def run_p2_cross_population(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],     # "young" | "elderly"
    model_name: str = "model",
    direction: str = "young_to_elderly",
) -> dict:
    """
    P2: Cross-population evaluation.

    direction : "young_to_elderly" — train on young, test on elderly
                "elderly_to_young" — reverse
    """
    groups_arr = np.array(groups)

    if direction == "young_to_elderly":
        train_mask = groups_arr == "young"
        test_mask  = groups_arr == "elderly"
    else:
        train_mask = groups_arr == "elderly"
        test_mask  = groups_arr == "young"

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    m = clone(model)
    y_pred, y_prob = _fit_predict(m, X_tr, y_tr, X_te)
    metrics = compute_metrics(y_te, y_pred, y_prob)
    metrics.update({
        "protocol":    "P2",
        "direction":   direction,
        "model":       model_name,
        "n_train":     int(train_mask.sum()),
        "n_test":      int(test_mask.sum()),
    })
    return metrics


def run_p3_cross_dataset(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    model_name: str = "model",
    train_dataset: str = "source",
    test_dataset:  str = "target",
) -> dict:
    """
    P3: Cross-dataset evaluation.

    Train and test sets come from independent datasets (different sensors, cohorts).
    """
    m = clone(model)
    y_pred, y_prob = _fit_predict(m, X_train, y_train, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics.update({
        "protocol":      "P3",
        "train_dataset": train_dataset,
        "test_dataset":  test_dataset,
        "model":         model_name,
        "n_train":       len(y_train),
        "n_test":        len(y_test),
    })
    return metrics


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fold-level results to mean ± std per model/protocol combination.
    """
    metric_cols = [
        "sensitivity", "specificity", "false_alarm_rate",
        "precision", "f1", "balanced_acc", "mcc",
    ]
    group_cols = [c for c in ["protocol", "direction", "model", "train_dataset", "test_dataset"]
                  if c in results_df.columns]

    agg = results_df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()
