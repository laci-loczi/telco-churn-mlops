import pytest

from mlops_project.config import Settings
from mlops_project.tracking import mlflow_available
from mlops_project.versioning import default_remote_url, pointer_md5, remote_object_uri


def test_pointer_md5_reads_the_dvc_pointer(tmp_path):
    pointer = tmp_path / "data.csv.dvc"
    pointer.write_text("outs:\n- md5: 0123456789abcdef0123456789abcdef\n  size: 10\n"
                       "  hash: md5\n  path: data.csv\n")
    assert pointer_md5(pointer) == "0123456789abcdef0123456789abcdef"
    assert pointer_md5(tmp_path / "missing.dvc") is None


def test_default_remote_url_reads_dvc_config(tmp_path):
    (tmp_path / ".dvc").mkdir()
    (tmp_path / ".dvc" / "config").write_text(
        "[core]\n    remote = storage\n"
        "['remote \"storage\"']\n    url = s3://bucket/store\n    endpointurl = http://x:1\n")
    assert default_remote_url(tmp_path) == "s3://bucket/store"


def test_remote_object_uri_uses_the_dvc3_layout():
    md5 = "ab" + "c" * 30
    assert remote_object_uri("s3://b/store/", md5) == f"s3://b/store/files/md5/ab/{'c' * 30}"
    assert remote_object_uri(None, md5) is None


def test_committed_dvc_config_has_no_credentials():
    config = Settings().project_root / ".dvc" / "config"
    if not config.exists():
        pytest.skip("DVC not initialised")
    text = config.read_text().lower()
    assert "access_key" not in text and "secret" not in text


@pytest.mark.live
def test_mlflow_server_is_reachable_when_the_stack_is_up():
    if not mlflow_available(Settings()):
        pytest.skip("MLflow stack not running")
    assert mlflow_available(Settings())
