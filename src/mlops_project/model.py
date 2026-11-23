"""The model: a scikit-learn Pipeline (preprocessing + estimator) and its metrics."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_model(train_params: dict, seed: int) -> Pipeline:
    family = train_params.get("family", "logreg")
    class_weight = None if train_params.get("class_weight", "none") == "none" else "balanced"
    if family == "logreg":
        estimator = LogisticRegression(
            C=float(train_params.get("C", 1.0)),
            max_iter=int(train_params.get("max_iter", 1000)),
            class_weight=class_weight,
        )
    elif family == "random_forest":
        max_depth = train_params.get("max_depth")
        estimator = RandomForestClassifier(
            n_estimators=int(train_params.get("n_estimators", 200)),
            max_depth=None if max_depth in (None, "none") else int(max_depth),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,  # deterministic and identical on every machine
        )
    else:
        raise ValueError(f"unknown model family {family!r}")

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def model_params(train_params: dict) -> dict:
    """The hyperparameters that actually apply to the chosen family (for MLflow)."""
    family = train_params.get("family", "logreg")
    keys = ["family", "class_weight"] + (
        ["C", "max_iter"] if family == "logreg" else ["n_estimators", "max_depth"]
    )
    return {k: train_params.get(k) for k in keys}


def evaluate_model(model: Pipeline, x_test, y_test) -> dict:
    """Binary metrics on the positive class (churn = 1)."""
    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    }
