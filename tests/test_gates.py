"""The gate is a pure function over dicts, so these need no files and no Docker."""

import json

import pytest

from mlops_project.gates import evaluate_gate
from mlops_project.registry import publish_decision

ACCEPTANCE = {
    "criteria_version": "1.0.0",
    "minimum": {"roc_auc": 0.80, "f1": 0.55, "recall": 0.70},
    "regression": {"primary_metric": "roc_auc", "max_drop": 0.01, "require_baseline": True},
    "slice_rules": {"min_support": 100, "min_positives": 30, "min_f1": 0.20, "max_f1_gap": 0.40},
    "fairness": {"senior_citizen": {"max_tpr_gap": 0.25}},
}
GOOD = {"roc_auc": 0.846, "f1": 0.628, "recall": 0.797, "precision": 0.52, "accuracy": 0.75}
BASELINE = {"source": {"alias": "staging"}, "metrics": dict(GOOD)}
SLICES = {
    "slices": {"contract": [
        {"group": "Month-to-month", "n": 968, "positives": 414, "f1": 0.66},
        {"group": "One year", "n": 366, "positives": 43, "f1": 0.26},
        {"group": "Two year", "n": 427, "positives": 10, "f1": 0.0},
    ]},
    "fairness": {"senior_citizen": {"tpr_gap": 0.198, "selection_rate_gap": 0.33}},
}


def gate(metrics=GOOD, slices=SLICES, baseline=BASELINE, acceptance=ACCEPTANCE):
    return evaluate_gate(metrics, slices, baseline, acceptance, run_id="run-1")


def status(report, name):
    return next(c["status"] for c in report["checks"] if c["name"] == name)


def test_the_current_model_is_a_go():
    report = gate()
    assert report["verdict"] == "GO"
    assert report["criteria_version"] == "1.0.0"
    # 3 floors + 1 regression + 2 scored slices + 1 fairness; Two year is skipped.
    assert report["summary"] == {"passed": 7, "failed": 0, "skipped": 1}


def test_a_floor_below_threshold_is_a_no_go():
    report = gate(metrics={**GOOD, "recall": 0.557})
    assert report["verdict"] == "NO-GO"
    assert status(report, "floor.recall") == "fail"
    assert "floor.recall" in report["reasons"]


def test_a_regression_beyond_tolerance_is_a_no_go():
    report = gate(metrics={**GOOD, "roc_auc": 0.83})   # drop 0.016 > 0.01
    assert status(report, "regression.roc_auc") == "fail"
    assert report["verdict"] == "NO-GO"


def test_a_drop_within_tolerance_passes():
    assert status(gate(metrics={**GOOD, "roc_auc": 0.84}), "regression.roc_auc") == "pass"


def test_a_missing_baseline_is_skipped_and_not_upgraded_to_a_pass():
    report = gate(baseline=None)
    assert status(report, "regression.roc_auc") == "skipped"
    assert report["summary"]["skipped"] == 2
    assert report["verdict"] == "NO-GO"          # require_baseline: true
    assert any("no baseline" in r for r in report["reasons"])


def test_without_require_baseline_the_skip_is_still_counted():
    rules = {**ACCEPTANCE, "regression": {**ACCEPTANCE["regression"], "require_baseline": False}}
    report = gate(baseline=None, acceptance=rules)
    assert report["verdict"] == "GO"
    assert report["summary"]["skipped"] == 2       # visible, never silently a pass


def test_a_small_slice_is_skipped_with_a_reason_not_passed():
    report = gate()
    check = next(c for c in report["checks"] if c["name"] == "slice.contract.Two year")
    assert check["status"] == "skipped"
    assert "too small" in check["reason"] and "positives=10" in check["reason"]


def test_a_collapsed_slice_fails():
    slices = json.loads(json.dumps(SLICES))
    slices["slices"]["contract"][1]["f1"] = 0.05
    report = gate(slices=slices)
    assert status(report, "slice.contract.One year") == "fail"
    assert report["verdict"] == "NO-GO"


def test_a_fairness_gap_above_its_limit_fails():
    slices = {**SLICES, "fairness": {"senior_citizen": {"tpr_gap": 0.31}}}
    assert status(gate(slices=slices), "fairness.senior_citizen.tpr_gap") == "fail"


def test_an_uncomputable_fairness_gap_is_skipped():
    slices = {**SLICES, "fairness": {"senior_citizen": {"tpr_gap": None}}}
    assert status(gate(slices=slices), "fairness.senior_citizen.tpr_gap") == "skipped"


def test_the_verdict_is_deterministic_and_has_no_timestamp():
    first = json.dumps(gate(), sort_keys=True)
    assert first == json.dumps(gate(), sort_keys=True)
    assert "time" not in first and "date" not in first


# ── publishing consults the verdict ─────────────────────────────────────────
@pytest.mark.parametrize("report, run_id, allowed", [
    ({"verdict": "GO", "run_id": "r1"}, "r1", True),
    ({"verdict": "NO-GO", "run_id": "r1", "reasons": ["floor.recall"]}, "r1", False),
    ({"verdict": "GO", "run_id": "old"}, "r1", False),   # a stale GO is not a GO
    (None, "r1", False),                                  # no verdict at all
    ({"verdict": "GO", "run_id": "r1"}, None, False),     # no model run
])
def test_publish_requires_a_go_for_this_exact_run(report, run_id, allowed):
    assert publish_decision(report, run_id)[0] is allowed
