import shutil
import subprocess

import pytest

from mlops_project.versioning import git_dirty


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "params.yaml").write_text("train:\n  C: 1.0\n")
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\n")
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return tmp_path


def test_a_clean_checkout_is_not_dirty(repo):
    assert git_dirty(repo) is False


def test_pipeline_bookkeeping_does_not_make_a_run_dirty(repo):
    """dvc repro rewrites dvc.lock mid-run; that is an output, not an input."""
    (repo / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n")
    assert git_dirty(repo) is False


def test_an_uncommitted_param_change_is_dirty(repo):
    (repo / "params.yaml").write_text("train:\n  C: 0.5\n")
    assert git_dirty(repo) is True
