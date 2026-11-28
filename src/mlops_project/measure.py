"""The measurements the acceptance thresholds are derived from (docs/hw2.md).

    uv run python src/main.py measure

1. What trivial constant predictors score on the test set (the floor must beat them).
2. How much the metrics move under resampling of the test set (bootstrap), which
   sets the regression tolerance: a drop smaller than the noise is not a signal.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score


def trivial_predictors(y_true) -> dict:
    y_true = np.asarray(y_true)
    out = {}
    for name, constant in (("always_stays", 0), ("always_churns", 1)):
        y_pred = np.full_like(y_true, constant)
        out[name] = {
            "accuracy": round(float((y_pred == y_true).mean()), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            # A constant score ranks nobody above anybody: ROC-AUC is 0.5.
            "roc_auc": 0.5,
        }
    return out


def bootstrap(y_true, y_pred, y_score, n: int = 1000, seed: int = 0) -> dict:
    """Std and 95% interval of each metric over n resamples of the test set."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    rng = np.random.default_rng(seed)
    draws = {"roc_auc": [], "f1": [], "recall": []}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        t, p, s = y_true[idx], y_pred[idx], y_score[idx]
        draws["roc_auc"].append(roc_auc_score(t, s))
        draws["f1"].append(f1_score(t, p, zero_division=0))
        draws["recall"].append(recall_score(t, p, zero_division=0))
    return {k: {"std": round(float(np.std(v)), 4),
                "p2.5": round(float(np.percentile(v, 2.5)), 4),
                "p97.5": round(float(np.percentile(v, 97.5)), 4)} for k, v in draws.items()}
