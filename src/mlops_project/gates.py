"""The model gate as one pure function over plain data structures.

    evaluate_gate(metrics, slices, baseline, acceptance, run_id) -> verdict

No file reads, no registry calls: `pipeline.gate` gathers the inputs and writes
the result. That is what makes the gate testable without Docker (tests/test_gates.py).

Every check has exactly one of three statuses:

    pass     evaluated, and it met the criterion
    fail     evaluated, and it did not  -> the verdict is NO-GO
    skipped  could not be evaluated (too few rows, no baseline, ...), with a reason

A skipped check never counts as a pass. It is counted in `summary.skipped` and
printed on the summary line; a skip the criteria mark as required (a missing
baseline with `require_baseline: true`) makes the verdict NO-GO.
"""

from __future__ import annotations

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"


def _r(value):
    return None if value is None else round(float(value), 4)


def _check(name: str, status: str, **fields) -> dict:
    return {"name": name, "status": status, **{k: _r(v) if isinstance(v, float) else v
                                               for k, v in fields.items()}}


def evaluate_gate(metrics: dict, slices: dict, baseline: dict | None, acceptance: dict,
                  run_id: str | None = None) -> dict:
    checks: list[dict] = []
    blocking: list[str] = []

    checks += _floor_checks(metrics, acceptance.get("minimum", {}))

    regression, missing_required = _regression_checks(metrics, baseline,
                                                      acceptance.get("regression", {}))
    checks += regression
    if missing_required:
        blocking.append("no baseline recorded, and regression.require_baseline is true")

    rules = acceptance.get("slice_rules", {})
    checks += _slice_checks(metrics, slices.get("slices", {}), rules)
    checks += _fairness_checks(slices.get("fairness", {}), acceptance.get("fairness", {}))

    passed = sum(c["status"] == PASS for c in checks)
    failed = sum(c["status"] == FAIL for c in checks)
    skipped = sum(c["status"] == SKIPPED for c in checks)
    reasons = [c["name"] for c in checks if c["status"] == FAIL] + blocking
    return {
        "verdict": "NO-GO" if reasons else "GO",
        "criteria_version": str(acceptance.get("criteria_version", "unversioned")),
        "run_id": run_id,
        "summary": {"passed": passed, "failed": failed, "skipped": skipped},
        "reasons": reasons,
        "checks": checks,
    }


def _floor_checks(metrics: dict, minimum: dict) -> list[dict]:
    out = []
    for metric in sorted(minimum):
        floor = float(minimum[metric])
        value = metrics.get(metric)
        name = f"floor.{metric}"
        if value is None:
            out.append(_check(name, SKIPPED, threshold=floor,
                              reason=f"metric {metric!r} not in metrics.json"))
        else:
            out.append(_check(name, PASS if value >= floor else FAIL,
                              observed=float(value), threshold=floor))
    return out


def _regression_checks(metrics: dict, baseline: dict | None, rules: dict):
    """Primary metric vs the champion; returns (checks, required_but_missing)."""
    if not rules:
        return [], False
    primary = rules["primary_metric"]
    max_drop = float(rules["max_drop"])
    name = f"regression.{primary}"
    base_metrics = (baseline or {}).get("metrics") or {}
    if primary not in base_metrics:
        reason = ("no baseline recorded (metrics/baseline.json missing or empty)"
                  if not base_metrics else f"baseline has no {primary!r}")
        return ([_check(name, SKIPPED, threshold=max_drop, reason=reason)],
                bool(rules.get("require_baseline", True)))
    if primary not in metrics:
        return [_check(name, SKIPPED, threshold=max_drop,
                       reason=f"metric {primary!r} not in metrics.json")], False
    delta = float(metrics[primary]) - float(base_metrics[primary])
    return [_check(name, PASS if delta >= -max_drop else FAIL,
                   observed=float(metrics[primary]), baseline=float(base_metrics[primary]),
                   delta=delta, threshold=max_drop)], False


def _slice_checks(metrics: dict, slices: dict, rules: dict) -> list[dict]:
    if not rules:
        return []
    min_support = int(rules["min_support"])
    min_positives = int(rules["min_positives"])
    min_f1 = float(rules["min_f1"])
    max_gap = float(rules["max_f1_gap"])
    aggregate_f1 = metrics.get("f1")
    out = []
    for axis in sorted(slices):
        for group in slices[axis]:
            name = f"slice.{axis}.{group['group']}"
            support = {"n": group["n"], "positives": group["positives"]}
            if group["n"] < min_support or group["positives"] < min_positives:
                out.append(_check(name, SKIPPED, **support, reason=(
                    f"too small to score: n={group['n']} (min {min_support}), "
                    f"positives={group['positives']} (min {min_positives})")))
                continue
            f1 = group["f1"]
            gap = None if aggregate_f1 is None else float(aggregate_f1) - float(f1)
            ok = f1 >= min_f1 and (gap is None or gap <= max_gap)
            out.append(_check(name, PASS if ok else FAIL, **support, f1=float(f1),
                              min_f1=min_f1, gap=gap, max_gap=max_gap))
    return out


def _fairness_checks(fairness: dict, rules: dict) -> list[dict]:
    out = []
    for axis in sorted(rules):
        for gap_name in sorted(rules[axis]):
            limit = float(rules[axis][gap_name])
            metric = gap_name.removeprefix("max_")  # max_tpr_gap -> tpr_gap
            name = f"fairness.{axis}.{metric}"
            observed = (fairness.get(axis) or {}).get(metric)
            if observed is None:
                out.append(_check(name, SKIPPED, threshold=limit,
                                  reason="fewer than two groups large enough to compare"))
            else:
                out.append(_check(name, PASS if observed <= limit else FAIL,
                                  observed=float(observed), threshold=limit))
    return out


def format_gate(report: dict) -> str:
    s = report["summary"]
    lines = [f"GATE {report['verdict']} (criteria {report['criteria_version']}): "
             f"{s['passed']} passed, {s['failed']} failed, {s['skipped']} SKIPPED"]
    for c in report["checks"]:
        detail = c.get("reason") or ", ".join(
            f"{k}={c[k]}" for k in ("observed", "baseline", "delta", "f1", "gap", "threshold")
            if c.get(k) is not None)
        lines.append(f"  {c['status'].upper():<8} {c['name']:<44} {detail}")
    for reason in report["reasons"]:
        lines.append(f"  -> NO-GO because: {reason}")
    return "\n".join(lines)
