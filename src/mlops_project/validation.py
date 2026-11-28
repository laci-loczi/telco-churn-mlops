"""Run a contract and turn the result into a deterministic verdict.

The verdict is committed (reports/validation.json, cache: false), so it holds no
timestamp and lists failures in a stable order: two runs on the same data give
byte-identical files.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

MAX_ROWS_LISTED = 10


def _value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def validate_frame(schema, frame: pd.DataFrame) -> dict:
    """Validate lazily (collect every failure) and summarise one entry per check.

    Pandera reports a row-level (dataframe) check once per column of a failing
    row; those are collapsed so the count is failing ROWS, not cells.
    """
    name = schema.__name__
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases.copy()
        frame_level = cases["schema_context"] == "DataFrameSchema"
        # Row checks have an index; schema checks (strict columns) do not and keep
        # their failure_case, which is the offending column's name.
        row_level = frame_level & cases["index"].notna()
        cases.loc[frame_level, "column"] = "<frame>"
        cases.loc[row_level, "failure_case"] = None
        cases["column"] = cases["column"].astype(str)
        cases["check"] = cases["check"].astype(str)
        cases["index"] = cases["index"].astype("Int64")
        cases = cases.drop_duplicates(["column", "check", "index"])
        checks = []
        for (column, check), group in sorted(cases.groupby(["column", "check"]),
                                             key=lambda item: item[0]):
            group = group.sort_values("index", na_position="first")
            checks.append({
                "column": column,
                "check": check,
                "failing_rows": int(len(group)),
                "first_rows": [_value(i) for i in group["index"].head(MAX_ROWS_LISTED)],
                "first_values": [_value(v) for v in group["failure_case"].head(MAX_ROWS_LISTED)],
            })
        return {"contract": name, "valid": False, "rows": len(frame),
                "failing_checks": len(checks), "failures": checks}
    return {"contract": name, "valid": True, "rows": len(frame), "failing_checks": 0,
            "failures": []}


def format_validation(report: dict) -> str:
    state = "PASSED" if report["valid"] else "FAILED"
    lines = [f"{report['contract']}: {state} on {report['rows']} rows"]
    for f in report["failures"]:
        values = f" values={f['first_values']}" if any(v is not None for v in f["first_values"]) else ""
        lines.append(f"  FAIL {f['column']:<16} {f['check']:<40} rows={f['failing_rows']} "
                     f"first={f['first_rows']}{values}")
    return "\n".join(lines)
