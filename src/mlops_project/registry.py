"""The model registry: register the pipeline's run, and walk the chain back."""

from __future__ import annotations

import json

from .config import Settings


class PublishRefused(RuntimeError):
    """The recorded gate verdict does not allow publishing this model."""


def publish_decision(gate_report: dict | None, run_id: str | None) -> tuple[bool, str]:
    """May the model of `run_id` be published? A pure function, tested directly.

    Only a GO recorded for THIS run counts: a GO left over from an earlier run
    says nothing about the model now in models/. The verdict is checked first,
    so a NO-GO is always the reason given when it applies.
    """
    if gate_report is None:
        return False, "no recorded verdict (reports/gate.json missing) — run `dvc repro`"
    if gate_report.get("verdict") != "GO":
        reasons = "; ".join(gate_report.get("reasons", [])) or "unknown"
        return False, f"gate verdict is {gate_report.get('verdict')}: {reasons}"
    if not run_id:
        return False, "models/run.json has no run_id (was MLflow up during `dvc repro`?)"
    if gate_report.get("run_id") != run_id:
        return False, (f"reports/gate.json is for run {gate_report.get('run_id')}, not the "
                       f"current model {run_id} — run `dvc repro`")
    return True, f"gate GO (criteria {gate_report.get('criteria_version')})"


def register(settings: Settings) -> dict:
    """Register `runs:/<run_id>/model` from models/run.json and point the alias at it.

    Refuses (PublishRefused) unless reports/gate.json is a GO for this run. The
    check comes before any registry call, so a refused model never gets a version.
    Registering from the run URI is what populates the version's `run_id`.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    run_id = json.loads(settings.run_info_path.read_text(encoding="utf-8")).get("run_id")
    gate_report = (json.loads(settings.gate_path.read_text(encoding="utf-8"))
                   if settings.gate_path.exists() else None)
    allowed, reason = publish_decision(gate_report, run_id)
    if not allowed:
        raise PublishRefused(f"refusing to register: {reason}")

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
            "already_registered": bool(existing), "reason": reason}


def refresh_baseline(settings: Settings) -> dict:
    """Write metrics/baseline.json from the model the ALIAS points at (not a version).

    No timestamp: the file is committed and must only change when the champion does.
    """
    from mlflow.tracking import MlflowClient

    from .pipeline import _write_json

    client = MlflowClient(settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(settings.registered_model_name,
                                                settings.model_alias)
    run = client.get_run(version.run_id)
    baseline = {
        "source": {"model": settings.registered_model_name, "alias": settings.model_alias,
                   "version": str(version.version), "run_id": version.run_id},
        "metrics": {k: round(float(v), 4) for k, v in sorted(run.data.metrics.items())},
    }
    _write_json(settings.baseline_path, baseline)
    return baseline


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
