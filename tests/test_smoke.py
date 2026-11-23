"""End-to-end on synthetic Telco-shaped data: clean -> split -> train -> evaluate.

No dataset, no Docker: these run on a fresh clone with nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_project.data import clean, features_and_target, split
from mlops_project.model import build_model, evaluate_model, model_params


def synthetic_telco(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18, 120, size=n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n)
    churn = np.where((contract == "Month-to-month") & (tenure < 24), "Yes", "No")
    churn[:10] = "Yes"  # make sure both classes exist
    yes_no = lambda: rng.choice(["Yes", "No"], size=n)  # noqa: E731
    addon = lambda: rng.choice(["Yes", "No", "No internet service"], size=n)  # noqa: E731
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # the real data's trap
    return pd.DataFrame({
        "customerID": [f"{i:04d}-XXXXX" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], size=n),
        "SeniorCitizen": rng.integers(0, 2, size=n),
        "Partner": yes_no(), "Dependents": yes_no(), "tenure": tenure,
        "PhoneService": yes_no(),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
        "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
        "Contract": contract, "PaperlessBilling": yes_no(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], size=n),
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
    })


@pytest.mark.parametrize("train_params", [
    {"family": "logreg", "C": 1.0, "max_iter": 1000, "class_weight": "none"},
    {"family": "logreg", "C": 0.5, "max_iter": 500, "class_weight": "balanced"},
    {"family": "random_forest", "n_estimators": 20, "max_depth": 4, "class_weight": "none"},
    {"family": "random_forest", "n_estimators": 20, "max_depth": "none", "class_weight": "none"},
])
def test_pipeline_trains_and_evaluates(train_params):
    df = clean(synthetic_telco())
    train_df, test_df = split(df, test_size=0.25, seed=0)
    model = build_model(train_params, seed=0)
    model.fit(*features_and_target(train_df))
    metrics = evaluate_model(model, *features_and_target(test_df))
    assert set(metrics) == {"roc_auc", "f1", "precision", "recall", "accuracy"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_model_params_only_reports_the_family_in_use():
    assert set(model_params({"family": "logreg", "C": 1.0})) == {
        "family", "class_weight", "C", "max_iter"}
    assert set(model_params({"family": "random_forest"})) == {
        "family", "class_weight", "n_estimators", "max_depth"}


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        build_model({"family": "svm"}, seed=0)
