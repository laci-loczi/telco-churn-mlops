"""The three DVC stages (prepare -> train -> evaluate), plus a hyperparameter sweep.

Each stage reads files and writes files, so DVC can hash both ends. MLflow is a
side channel: if the server is unreachable the stages still produce their
outputs, and `models/run.json` records that no run was logged.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from .config import Settings, load_params
from .data import clean, features_and_target, load_raw, split
from .model import build_model, evaluate_model, model_params
from .tracking import lineage_tags, mlflow_available, use_experiment


def _write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": byte-identical on every OS, so dvc.lock's md5 holds.
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _read_split(settings: Settings, name: str) -> pd.DataFrame:
    return pd.read_csv(settings.processed_dir / f"{name}.csv",
                       dtype={"SeniorCitizen": str})


# ── stage 1: prepare ─────────────────────────────────────────────────────────
def prepare(settings: Settings) -> dict:
    params = load_params(settings)
    df = clean(load_raw(settings.data_path))
    train_df, test_df = split(df, params["prepare"]["test_size"], params["random_seed"])
    _write_csv(train_df, settings.processed_dir / "train.csv")
    _write_csv(test_df, settings.processed_dir / "test.csv")
    return {"rows": len(df), "train": len(train_df), "test": len(test_df),
            "churn_rate": round(float(df["Churn"].mean()), 4)}


# ── stage 2: train ───────────────────────────────────────────────────────────
def _fit(settings: Settings, train_params: dict, seed: int):
    x_train, y_train = features_and_target(_read_split(settings, "train"))
    model = build_model(train_params, seed)
    model.fit(x_train, y_train)
    return model, x_train


def _log_run(settings: Settings, model, x_train, train_params: dict, seed: int,
             run_type: str, metrics: dict | None = None) -> str:
    """Log one MLflow run (params, lineage tags, model, optional metrics)."""
    import mlflow
    import mlflow.sklearn

    use_experiment(settings)
    name = f"{train_params['family']}-{run_type}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({**model_params(train_params), "random_seed": seed})
        mlflow.set_tags({**lineage_tags(settings), "run_type": run_type})
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", input_example=x_train.head(5))
        return run.info.run_id


def train(settings: Settings) -> dict:
    params = load_params(settings)
    seed = params["random_seed"]
    model, x_train = _fit(settings, params["train"], seed)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_path)

    run_id = None
    if mlflow_available(settings):
        run_id = _log_run(settings, model, x_train, params["train"], seed, run_type="pipeline")
    _write_json(settings.run_info_path, {
        "run_id": run_id,
        "experiment": settings.experiment_name if run_id else None,
    })
    return {"run_id": run_id, "family": params["train"]["family"]}


# ── stage 3: evaluate ────────────────────────────────────────────────────────
def evaluate(settings: Settings) -> dict:
    model = joblib.load(settings.model_path)
    x_test, y_test = features_and_target(_read_split(settings, "test"))
    metrics = evaluate_model(model, x_test, y_test)
    _write_json(settings.metrics_path, metrics)

    run_id = json.loads(settings.run_info_path.read_text(encoding="utf-8")).get("run_id")
    if run_id and mlflow_available(settings):
        from mlflow.tracking import MlflowClient

        client = MlflowClient(settings.mlflow_tracking_uri)
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
    return {"metrics": metrics, "run_id": run_id}


# ── hyperparameter sweep (outside DVC: comparison runs, not the pipeline) ────
SWEEP_GRID = [
    {"family": "logreg", "C": 0.1, "max_iter": 1000, "class_weight": "none"},
    {"family": "logreg", "C": 1.0, "max_iter": 1000, "class_weight": "none"},
    {"family": "logreg", "C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    {"family": "logreg", "C": 10.0, "max_iter": 1000, "class_weight": "none"},
    {"family": "random_forest", "n_estimators": 200, "max_depth": 8, "class_weight": "none"},
    {"family": "random_forest", "n_estimators": 300, "max_depth": "none", "class_weight": "none"},
]


def sweep(settings: Settings, grid: list[dict] | None = None) -> list[dict]:
    """Train and log one run per grid point on the prepared split."""
    if not mlflow_available(settings):
        raise RuntimeError(f"MLflow not reachable at {settings.mlflow_tracking_uri} — "
                           "start the stack with `docker compose up -d --wait`.")
    params = load_params(settings)
    seed = params["random_seed"]
    x_test, y_test = features_and_target(_read_split(settings, "test"))
    results = []
    for point in grid or SWEEP_GRID:
        model, x_train = _fit(settings, point, seed)
        metrics = evaluate_model(model, x_test, y_test)
        run_id = _log_run(settings, model, x_train, point, seed, "sweep", metrics)
        results.append({"run_id": run_id, **model_params(point), **metrics})
    return results
