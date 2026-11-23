"""Loading and cleaning the Telco churn data.

The cleaning rules encode the domain notes in docs/DATA_DICTIONARY.md:

- `TotalCharges` is numeric but stored as text, with blanks for the customers
  whose `tenure` is 0. They have never been billed, so their total is 0, not a
  mean to impute.
- `customerID` is an identifier: no signal, leakage risk, dropped.
- `Churn` Yes/No becomes 1/0.
- "No internet service" / "No phone service" are real categories, left as-is.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ID_COLUMN, TARGET_COLUMN

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is 0/1 while every other binary is Yes/No; treat it as a
# category so it is encoded like its neighbours rather than scaled.
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Fetch it with `dvc pull` (see README), or rebuild it with "
            "`uv run python src/main.py make-data --version 2`."
        )
    # keep_default_na=False: read blanks as "" so the TotalCharges rule below is
    # explicit, instead of pandas silently turning them into NaN.
    return pd.read_csv(path, dtype={"TotalCharges": str}, keep_default_na=False)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", None),
                          errors="coerce")
    # Blank TotalCharges <=> tenure 0: never billed, so the total is 0.
    total = total.where(~(total.isna() & (df["tenure"] == 0)), 0.0)
    if total.isna().any():
        bad = int(total.isna().sum())
        raise ValueError(f"{bad} rows have an unparseable TotalCharges with tenure > 0.")
    df["TotalCharges"] = total.astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Churn must be 'Yes' or 'No'.")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df.drop(columns=[ID_COLUMN], errors="ignore")[FEATURES + [TARGET_COLUMN]]


def split(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified train/test split, so both halves keep the ~26% churn rate."""
    return train_test_split(df, test_size=test_size, random_state=seed,
                            stratify=df[TARGET_COLUMN])


def features_and_target(df: pd.DataFrame):
    return df[FEATURES], df[TARGET_COLUMN]
