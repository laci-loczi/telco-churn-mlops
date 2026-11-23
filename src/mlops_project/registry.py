"""The model registry: register the pipeline's run, and walk the chain back."""

from __future__ import annotations

import json

from .config import Settings


def register(settings: Settings) -> dict:
    """Register `runs:/<run_id>/model` from models/run.json and point the alias at it.

    Registering from the run URI is what populates the version's `run_id` — the
    link from the registry back to the run.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    run_id = json.loads(settings.run_info_path.read_text(encoding="utf-8")).get("run_id")
    if not run_id:
        raise RuntimeError("models/run.json has no run_id — run `dvc repro` with MLflow up first.")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(settings.mlflow_tracking_uri)
    existing = [v for v in client.search_model_versions(f"name='{settings.registered_model_name}'")
                if v.run_id == run_id]
    if existing:
        version = existing[0]
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", settings.registered_model_name)
    client.set_registered_model_alias(settings.registered_model_name, settings.model_alias,
                                      version.version)
    return {"name": settings.registered_model_name, "version": version.version,
            "alias": settings.model_alias, "run_id": version.run_id,
            "already_registered": bool(existing)}


def chain(settings: Settings) -> dict:
    """alias -> version -> run -> params/metrics -> git commit -> data md5 -> remote URI."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(settings.registered_model_name,
                                                settings.model_alias)
    run = client.get_run(version.run_id)
    tags = run.data.tags
    return {
        "alias": f"{settings.registered_model_name}@{settings.model_alias}",
        "version": version.version,
        "run_id": version.run_id,
        "run_name": run.info.run_name,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "git_commit": tags.get("git_commit"),
        "git_dirty": tags.get("git_dirty"),
        "dvc_md5": tags.get("dvc_md5"),
        "data_remote_uri": tags.get("data_remote_uri"),
    }
