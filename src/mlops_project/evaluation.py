"""Per-slice metrics and rate gaps.

Slices are computed on the RAW slice columns of the test set. `clean()` only
rewrites TotalCharges (blank -> 0), maps Churn to 0/1 and turns SeniorCitizen
into a string; the model's scaling and one-hot encoding happen inside the
sklearn Pipeline, so `data/processed/test.csv` still holds raw Contract,
tenure, SeniorCitizen and gender values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _r(value) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 4)


def slice_labels(frame: pd.DataFrame, definition: dict) -> pd.Series:
    """Map each row to its group label for one slice definition."""
    column = frame[definition["column"]]
    if "bins" in definition:
        edges = definition["bins"]
        # Right-inclusive bins starting just below the first edge: [0,12,48,72]
        # gives 0-12, 13-48, 49-72.
        return pd.cut(column, bins=[edges[0] - 1, *edges[1:]], labels=definition["labels"],
                      right=True).astype(str)
    labels = definition.get("labels")
    values = column.astype(str)
    return values.map(labels).fillna(values) if labels else values


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = int(len(y_true))
    positives = int(y_true.sum())
    predicted = int(y_pred.sum())
    row = {"n": n, "positives": positives, "predicted_positive": predicted,
           "selection_rate": _r(predicted / n) if n else None}
    if positives == 0:
        row.update(f1=None, precision=None, recall=None, tpr=None)
        return row
    row.update(
        f1=_r(f1_score(y_true, y_pred, zero_division=0)),
        precision=_r(precision_score(y_true, y_pred, zero_division=0)),
        recall=_r(recall_score(y_true, y_pred, zero_division=0)),
    )
    row["tpr"] = row["recall"]
    return row


def slice_report(frame: pd.DataFrame, y_true, y_pred, definitions: dict) -> dict:
    """{axis: [{group, n, positives, ...}, ...]} with groups in a stable order."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    report = {}
    for axis in sorted(definitions):
        labels = slice_labels(frame, definitions[axis]).to_numpy()
        order = definitions[axis].get("order") or sorted(set(labels))
        if isinstance(order, dict):
            order = list(order.values())
        report[axis] = [
            {"group": group, **group_metrics(y_true[labels == group], y_pred[labels == group])}
            for group in order if (labels == group).any()
        ]
    return report


def rate_gaps(groups: list[dict], min_support: int, min_positives: int) -> dict:
    """Selection-rate and TPR gaps (max - min) over groups large enough to score."""
    eligible = [g for g in groups if g["n"] >= min_support and g["positives"] >= min_positives]
    skipped = sorted(g["group"] for g in groups if g not in eligible)
    if len(eligible) < 2:
        return {"selection_rate_gap": None, "tpr_gap": None, "groups_compared": [],
                "groups_skipped": skipped}
    sel = [g["selection_rate"] for g in eligible]
    tpr = [g["tpr"] for g in eligible]
    return {
        "selection_rate_gap": _r(max(sel) - min(sel)),
        "tpr_gap": _r(max(tpr) - min(tpr)),
        "groups_compared": sorted(g["group"] for g in eligible),
        "groups_skipped": skipped,
    }
