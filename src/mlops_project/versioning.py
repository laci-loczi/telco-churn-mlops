"""Where a run's inputs came from: the data version (DVC) and the code version (Git).

These are the values that get stamped on every MLflow run, so that a registered
model can be followed back to the exact bytes it was trained on.
"""

from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

import yaml


def pointer_md5(pointer_path: Path) -> str | None:
    """The md5 recorded in a `.dvc` pointer file — the dataset's version id."""
    if not pointer_path.exists():
        return None
    pointer = yaml.safe_load(pointer_path.read_text(encoding="utf-8")) or {}
    outs = pointer.get("outs") or []
    return outs[0].get("md5") if outs else None


def default_remote_url(project_root: Path) -> str | None:
    """The URL of the default DVC remote, read from the committed .dvc/config."""
    config_path = project_root / ".dvc" / "config"
    if not config_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    name = parser.get("core", "remote", fallback=None)
    if not name:
        return None
    section = f"'remote \"{name}\"'"
    return parser.get(section, "url", fallback=None)


def remote_object_uri(remote_url: str | None, md5: str | None) -> str | None:
    """Where DVC 3 stores an object with this md5 in the remote.

    DVC 3 lays the remote out as files/md5/<first 2 hex chars>/<remaining 30>.
    """
    if not remote_url or not md5:
        return None
    return f"{remote_url.rstrip('/')}/files/md5/{md5[:2]}/{md5[2:]}"


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_dirty(project_root: Path) -> bool:
    """True when tracked files differ from HEAD (untracked files are ignored)."""
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=project_root, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())
