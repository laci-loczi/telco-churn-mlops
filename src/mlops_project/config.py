"""Project configuration.

Two sources, deliberately separate:

- `.env` holds *where things are* (service URLs, names). It is git-ignored and
  differs per machine.
- `params.yaml` holds *what the model is* (hyperparameters). It is committed and
  DVC hashes it, so a change makes the right stages stale. DVC cannot hash an
  environment variable, which is why no hyperparameter lives in `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    mlflow_tracking_uri: str = "http://127.0.0.1:5600"
    experiment_name: str = "telco-churn"
    registered_model_name: str = "telco-churn-classifier"
    model_alias: str = "staging"

    # ── paths (all relative to the project root) ──
    @property
    def data_path(self) -> Path:
        """The DVC-tracked dataset."""
        return self.project_root / "data" / "telco_churn.csv"

    @property
    def pointer_path(self) -> Path:
        return self.project_root / "data" / "telco_churn.csv.dvc"

    @property
    def source_path(self) -> Path:
        """The raw IBM download the dataset versions are built from (git-ignored)."""
        return self.project_root / "data" / "source" / "Telco-Customer-Churn.csv"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def model_path(self) -> Path:
        return self.project_root / "models" / "model.joblib"

    @property
    def run_info_path(self) -> Path:
        return self.project_root / "models" / "run.json"

    @property
    def metrics_path(self) -> Path:
        return self.project_root / "metrics" / "metrics.json"

    @property
    def slices_path(self) -> Path:
        return self.project_root / "metrics" / "slices.json"

    @property
    def baseline_path(self) -> Path:
        """The recorded champion's metrics (committed; refreshed from the registry)."""
        return self.project_root / "metrics" / "baseline.json"

    @property
    def validation_path(self) -> Path:
        return self.project_root / "reports" / "validation.json"

    @property
    def gate_path(self) -> Path:
        return self.project_root / "reports" / "gate.json"

    @property
    def params_path(self) -> Path:
        return self.project_root / "params.yaml"


def load_settings(project_root: Path | None = None) -> Settings:
    root = project_root or PROJECT_ROOT
    load_dotenv(root / ".env")
    return Settings(
        project_root=root,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5600"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn"),
        registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "telco-churn-classifier"),
        model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "staging"),
    )


def load_params(settings: Settings) -> dict:
    """Read params.yaml and validate the parts the pipeline depends on."""
    params = yaml.safe_load(settings.params_path.read_text(encoding="utf-8")) or {}
    validate_params(params)
    return params


def validate_params(params: dict) -> None:
    test_size = params.get("prepare", {}).get("test_size", 0.25)
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("prepare.test_size must be between 0 and 1 (exclusive).")
    train = params.get("train", {})
    family = train.get("family", "logreg")
    if family not in ("logreg", "random_forest"):
        raise ValueError(f"train.family must be 'logreg' or 'random_forest', got {family!r}.")
    if float(train.get("C", 1.0)) <= 0:
        raise ValueError("train.C must be greater than 0.")
    if int(train.get("max_iter", 1000)) <= 0:
        raise ValueError("train.max_iter must be greater than 0.")
    if train.get("class_weight", "none") not in ("none", "balanced"):
        raise ValueError("train.class_weight must be 'none' or 'balanced'.")
