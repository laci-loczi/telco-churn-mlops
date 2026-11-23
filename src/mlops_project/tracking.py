"""MLflow helpers: is the server up, which tags every run carries, and the
runs-for-data query."""

from __future__ import annotations

import urllib.request

from .config import Settings
from .versioning import default_remote_url, git_commit, git_dirty, pointer_md5, remote_object_uri


def mlflow_available(settings: Settings, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{settings.mlflow_tracking_uri}/health", timeout=timeout):
            return True
    except Exception:  # noqa: BLE001 -- any failure means "not reachable"
        return False


def lineage_tags(settings: Settings) -> dict:
    """The tags that link a run to its code and its data."""
    md5 = pointer_md5(settings.pointer_path)
    remote = default_remote_url(settings.project_root)
    return {
        "git_commit": git_commit(settings.project_root) or "unknown",
        "git_dirty": str(git_dirty(settings.project_root)).lower(),
        "dvc_md5": md5 or "untracked",
        "dvc_remote": remote or "none",
        "data_remote_uri": remote_object_uri(remote, md5) or "none",
        "data_path": "data/telco_churn.csv",
    }


def use_experiment(settings: Settings) -> None:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)


def search_runs_by_data_version(settings: Settings, md5: str):
    """Every run in the experiment that was trained on data with this md5."""
    import mlflow

    use_experiment(settings)
    return mlflow.search_runs(
        experiment_names=[settings.experiment_name],
        filter_string=f"tags.dvc_md5 = '{md5}'",
        order_by=["attributes.start_time ASC"],
    )
