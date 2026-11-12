"""
Classical ML models for fall detection.

Models: Random Forest, SVM (RBF), XGBoost.
All use the same scikit-learn Pipeline pattern:
    StandardScaler → Classifier
so they can be swapped interchangeably in the evaluation loop.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def build_classical_models(random_state: int = 42) -> dict[str, Pipeline]:
    """
    Return a dict of named scikit-learn Pipelines.

    Each pipeline: StandardScaler → Classifier (predict_proba supported).
    """
    models: dict[str, Pipeline] = {}

    # Random Forest — interpretable, robust to small datasets
    models["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    # SVM (RBF) — strong baseline for time-series features (Özdemir & Barshan 2014)
    models["svm"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )),
    ])

    # XGBoost — gradient boosting, allows SHAP feature importance
    if _HAS_XGB:
        models["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # adjust for class imbalance per experiment
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
    else:
        print("[warn] xgboost not installed — XGBoost model skipped.")

    return models
