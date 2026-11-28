import pandas as pd
import pytest

from mlops_project.data import clean
from mlops_project.evaluation import rate_gaps, slice_report
from mlops_project.schemas import ADDON_COLUMNS, IngestionSchema, ModelInputSchema
from mlops_project.validation import validate_frame

from .test_smoke import synthetic_telco


@pytest.fixture
def raw():
    """Synthetic rows that respect every cross-field rule of the real data."""
    df = synthetic_telco(n=300)
    no_internet = df["InternetService"] == "No"
    for column in ADDON_COLUMNS:
        df.loc[no_internet, column] = "No internet service"
        df.loc[~no_internet & (df[column] == "No internet service"), column] = "No"
    no_phone = df["PhoneService"] == "No"
    df.loc[no_phone, "MultipleLines"] = "No phone service"
    df.loc[~no_phone & (df["MultipleLines"] == "No phone service"), "MultipleLines"] = "No"
    return df


def failing(report):
    return {(f["column"], f["check"]) for f in report["failures"]}


def test_clean_synthetic_data_passes_both_contracts(raw):
    assert validate_frame(IngestionSchema, raw)["valid"]
    assert validate_frame(ModelInputSchema, clean(raw))["valid"]


def test_ingestion_rejects_impossible_values(raw):
    raw.loc[0, "tenure"] = -3
    raw.loc[1, "MonthlyCharges"] = 5695.0
    report = validate_frame(IngestionSchema, raw)
    assert not report["valid"]
    assert ("tenure", "greater_than_or_equal_to(0)") in failing(report)
    assert ("MonthlyCharges", "less_than_or_equal_to(150.0)") in failing(report)


def test_ingestion_rejects_inconsistent_rows(raw):
    internet = raw.index[raw["InternetService"] != "No"][0]
    raw.loc[internet, "InternetService"] = "No"
    report = validate_frame(IngestionSchema, raw)
    assert ("<frame>", "no_internet_means_no_addons") in failing(report)


def test_ingestion_is_strict_about_columns(raw):
    assert not validate_frame(IngestionSchema, raw.assign(extra=1))["valid"]
    assert not validate_frame(IngestionSchema, raw.drop(columns=["gender"]))["valid"]


def test_model_input_is_stricter_than_ingestion(raw):
    cleaned = clean(raw)
    # The identifier must not reach the model (leakage).
    assert not validate_frame(ModelInputSchema, cleaned.assign(customerID="0000-AAAAA"))["valid"]
    # A blank TotalCharges is fine on arrival, but never in model input.
    assert not validate_frame(ModelInputSchema,
                              cleaned.assign(TotalCharges=float("nan")))["valid"]


def test_the_validation_verdict_is_byte_stable(raw):
    raw.loc[0, "tenure"] = -3
    assert validate_frame(IngestionSchema, raw) == validate_frame(IngestionSchema, raw)


def test_slice_rows_carry_n_and_positives():
    frame = pd.DataFrame({"tenure": [1, 5, 20, 30, 60, 70],
                          "SeniorCitizen": ["0", "1", "0", "1", "0", "1"]})
    y_true = [1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1]
    definitions = {"tenure_band": {"column": "tenure", "bins": [0, 12, 48, 72],
                                   "labels": ["0-12m", "13-48m", "49-72m"],
                                   "order": ["0-12m", "13-48m", "49-72m"]}}
    rows = slice_report(frame, y_true, y_pred, definitions)["tenure_band"]
    assert [(r["group"], r["n"], r["positives"]) for r in rows] == [
        ("0-12m", 2, 2), ("13-48m", 2, 1), ("49-72m", 2, 0)]
    assert rows[2]["f1"] is None    # no positives: F1 is undefined, not 0


def test_rate_gaps_ignore_groups_too_small_to_score():
    groups = [{"group": "a", "n": 500, "positives": 100, "selection_rate": 0.4, "tpr": 0.8},
              {"group": "b", "n": 400, "positives": 90, "selection_rate": 0.3, "tpr": 0.7},
              {"group": "tiny", "n": 9, "positives": 2, "selection_rate": 1.0, "tpr": 0.0}]
    gaps = rate_gaps(groups, min_support=100, min_positives=30)
    assert gaps["tpr_gap"] == 0.1 and gaps["groups_skipped"] == ["tiny"]
