import pytest

from mlops_project.config import Settings, load_params, validate_params


def test_defaults_use_the_56xx_ports():
    assert Settings().mlflow_tracking_uri.endswith(":5600")


def test_committed_params_are_valid():
    params = load_params(Settings())
    assert params["train"]["family"] in ("logreg", "random_forest")


@pytest.mark.parametrize("bad", [
    {"prepare": {"test_size": 1.5}},
    {"train": {"family": "svm"}},
    {"train": {"C": 0}},
    {"train": {"max_iter": -1}},
    {"train": {"class_weight": "auto"}},
])
def test_invalid_params_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_params(bad)
