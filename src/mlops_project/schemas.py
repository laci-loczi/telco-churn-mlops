"""The two data contracts.

- `IngestionSchema`: the CSV as it arrives, before anything touches it. It is
  checked by the `validate` stage, and `prepare` depends on its verdict, so a
  violation stops the pipeline before any work is done.
- `ModelInputSchema`: what the model is allowed to see, after `clean()`. It is
  checked in `prepare`, right before the train/test split is written.

Every bound is derived from the IBM data and the business, not copied; see
docs/hw2.md for the measured numbers behind each one.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]
ADDON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies"]

# Tenure: the IBM extract covers at most 72 months (6 years); a longer tenure
# means a different extract or a corrupted value.
MAX_TENURE = 72
# Monthly charge: the cheapest plan (phone only) is 18.25 and the full bundle
# 118.75. 15 / 150 leave headroom for a price change, but reject cents-vs-dollars
# or sign errors.
MIN_MONTHLY, MAX_MONTHLY = 15.0, 150.0
# TotalCharges / (tenure * MonthlyCharges) is 0.69–1.57 in the real data: the
# current price differs from what was paid in earlier months. Outside 0.5–2.0
# the two columns no longer describe the same customer.
MIN_TOTAL_RATIO, MAX_TOTAL_RATIO = 0.5, 2.0


class IngestionSchema(pa.DataFrameModel):
    """The raw file, read with `TotalCharges` as text and blanks kept as ''."""

    customerID: Series[str] = pa.Field(str_matches=r"^\d{4}-[A-Z]{5}$", unique=True)
    gender: Series[str] = pa.Field(isin=["Female", "Male"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=YES_NO)
    Dependents: Series[str] = pa.Field(isin=YES_NO)
    tenure: Series[int] = pa.Field(ge=0, le=MAX_TENURE)
    PhoneService: Series[str] = pa.Field(isin=YES_NO)
    MultipleLines: Series[str] = pa.Field(isin=["Yes", "No", "No phone service"])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(isin=ADDON)
    OnlineBackup: Series[str] = pa.Field(isin=ADDON)
    DeviceProtection: Series[str] = pa.Field(isin=ADDON)
    TechSupport: Series[str] = pa.Field(isin=ADDON)
    StreamingTV: Series[str] = pa.Field(isin=ADDON)
    StreamingMovies: Series[str] = pa.Field(isin=ADDON)
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=YES_NO)
    PaymentMethod: Series[str] = pa.Field(isin=[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ])
    MonthlyCharges: Series[float] = pa.Field(ge=MIN_MONTHLY, le=MAX_MONTHLY)
    # Text on arrival: a number, or a blank for a customer never billed.
    TotalCharges: Series[str] = pa.Field(str_matches=r"^\s*$|^\d+(\.\d+)?$")
    Churn: Series[str] = pa.Field(isin=YES_NO)

    class Config:
        strict = True   # an unexpected or missing column is a different file
        coerce = False  # a dtype surprise is a finding, not something to fix silently

    @pa.dataframe_check
    def blank_total_only_for_new_customers(cls, df):
        """TotalCharges is blank exactly when tenure is 0 (never billed)."""
        blank = df["TotalCharges"].str.strip() == ""
        return blank == (df["tenure"] == 0)

    @pa.dataframe_check
    def total_matches_tenure_times_monthly(cls, df):
        """TotalCharges ~ tenure * MonthlyCharges, within the 0.5–2.0 band."""
        total = pd.to_numeric(df["TotalCharges"].str.strip().replace("", None), errors="coerce")
        expected = df["tenure"] * df["MonthlyCharges"]
        ratio = total / expected.where(expected > 0)
        return ratio.isna() | ratio.between(MIN_TOTAL_RATIO, MAX_TOTAL_RATIO)

    @pa.dataframe_check
    def no_internet_means_no_addons(cls, df):
        """An add-on is 'No internet service' exactly when InternetService is 'No'."""
        no_internet = df["InternetService"] == "No"
        checks = [(df[column] == "No internet service") == no_internet
                  for column in ADDON_COLUMNS]
        return pd.concat(checks, axis=1).all(axis=1)

    @pa.dataframe_check
    def no_phone_means_no_lines(cls, df):
        return (df["MultipleLines"] == "No phone service") == (df["PhoneService"] == "No")


class ModelInputSchema(pa.DataFrameModel):
    """The cleaned frame `prepare` splits: features + the 0/1 target, nothing else."""

    gender: Series[str] = pa.Field(isin=["Female", "Male"])
    # Stricter: a string category, not 0/1. If it reached the model as an int it
    # would be scaled as a number while its 15 Yes/No neighbours are one-hot encoded.
    SeniorCitizen: Series[str] = pa.Field(isin=["0", "1"])
    Partner: Series[str] = pa.Field(isin=YES_NO)
    Dependents: Series[str] = pa.Field(isin=YES_NO)
    tenure: Series[int] = pa.Field(ge=0, le=MAX_TENURE)
    PhoneService: Series[str] = pa.Field(isin=YES_NO)
    MultipleLines: Series[str] = pa.Field(isin=["Yes", "No", "No phone service"])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(isin=ADDON)
    OnlineBackup: Series[str] = pa.Field(isin=ADDON)
    DeviceProtection: Series[str] = pa.Field(isin=ADDON)
    TechSupport: Series[str] = pa.Field(isin=ADDON)
    StreamingTV: Series[str] = pa.Field(isin=ADDON)
    StreamingMovies: Series[str] = pa.Field(isin=ADDON)
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=YES_NO)
    PaymentMethod: Series[str] = pa.Field(isin=[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ])
    MonthlyCharges: Series[float] = pa.Field(ge=MIN_MONTHLY, le=MAX_MONTHLY)
    # Stricter: numeric and never null. Ingestion allows a blank; the model cannot
    # take one, and the blank -> 0 rule must already have run.
    TotalCharges: Series[float] = pa.Field(ge=0.0, le=MAX_TENURE * MAX_MONTHLY * MAX_TOTAL_RATIO,
                                           nullable=False)
    Churn: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        # Stricter: strict means customerID cannot be here. An identifier the model
        # can see is a leakage path (it memorises customers, not behaviour).
        strict = True
        coerce = False
