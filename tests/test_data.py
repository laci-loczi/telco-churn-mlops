import pytest

from mlops_project.data import FEATURES, clean, split
from mlops_project.datasets import build_version

from .test_smoke import synthetic_telco


def test_blank_total_charges_become_zero_for_new_customers():
    raw = synthetic_telco()
    cleaned = clean(raw)
    new = raw["tenure"] == 0
    assert new.any()
    assert (cleaned.loc[new, "TotalCharges"] == 0.0).all()
    assert cleaned["TotalCharges"].dtype == float


def test_blank_total_charges_with_tenure_is_an_error():
    raw = synthetic_telco()
    raw.loc[raw["tenure"] > 0, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="TotalCharges"):
        clean(raw)


def test_clean_drops_the_id_and_encodes_the_target():
    cleaned = clean(synthetic_telco())
    assert "customerID" not in cleaned.columns
    assert list(cleaned.columns) == FEATURES + ["Churn"]
    assert set(cleaned["Churn"].unique()) <= {0, 1}


def test_split_is_stratified_and_deterministic():
    df = clean(synthetic_telco(n=400))
    a_train, a_test = split(df, 0.25, seed=42)
    b_train, b_test = split(df, 0.25, seed=42)
    assert a_test.index.equals(b_test.index)
    assert abs(a_train["Churn"].mean() - a_test["Churn"].mean()) < 0.05


def test_dataset_versions_are_byte_stable(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"h1,h2\r\n" + b"".join(f"{i},x\r\n".encode() for i in range(6000)))
    v1, v2 = tmp_path / "v1.csv", tmp_path / "v2.csv"
    assert build_version(1, source, v1) == 5000
    assert build_version(2, source, v2) == 6000
    assert b"\r" not in v1.read_bytes()
    again = tmp_path / "again.csv"
    build_version(1, source, again)
    assert again.read_bytes() == v1.read_bytes()
