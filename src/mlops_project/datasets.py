"""Building the dataset versions from the raw IBM download.

The IBM file is a static snapshot, so the "data changes over time" story is a
batch simulation (the course's convention):

- version 1: the first 5,000 customers, the initial customer base
- version 2: all 7,043 customers, i.e. v1 plus a batch of 2,043 new signups

The rows are written byte-for-byte as they appear in the source (no parsing), so
building the same version twice gives the same md5 on any machine.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
VERSION_ROWS = {1: 5000, 2: None}  # None = every row


def download_source(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def build_version(version: int, source: Path, out: Path) -> int:
    """Write dataset `version` to `out` and return its number of data rows."""
    if version not in VERSION_ROWS:
        raise ValueError(f"unknown dataset version {version}; choose from {sorted(VERSION_ROWS)}")
    lines = source.read_bytes().replace(b"\r\n", b"\n").splitlines()
    header, rows = lines[0], [line for line in lines[1:] if line.strip()]
    limit = VERSION_ROWS[version]
    selected = rows if limit is None else rows[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\n".join([header, *selected]) + b"\n")
    return len(selected)
