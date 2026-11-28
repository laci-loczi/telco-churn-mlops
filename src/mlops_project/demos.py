"""The two gate demonstrations (docs/hw2.md), runnable verbatim:

    uv run python src/main.py demo-data-gate    # R2: bad rows -> dvc repro refuses
    uv run python src/main.py demo-model-gate   # R6: worse model -> gate + register refuse

Each one plants a fault, shows the refusal, and restores the COMMITTED state
exactly (from Git and the DVC cache, so nothing is retrained). They refuse to
start on a working tree with uncommitted pipeline changes, because the restore
would throw those away.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys

import pandas as pd

from .config import Settings

GUARDED = ["params.yaml", "dvc.lock", "models/run.json", "metrics", "reports",
           "data/telco_churn.csv.dvc"]


def _banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def _run(settings: Settings, *args: str) -> int:
    """Run a command in the project root, streaming its output; return the exit code."""
    print(f"$ {' '.join(args)}", flush=True)
    executable = [sys.executable, "-m", "dvc"] if args[0] == "dvc" else [sys.executable]
    command = [*executable, *args[1:]]
    result = subprocess.run(command, cwd=settings.project_root, check=False)
    print(f"[exit code {result.returncode}]", flush=True)
    return result.returncode


def _require_clean(settings: Settings) -> None:
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=settings.project_root, capture_output=True, text=True,
                            check=False).stdout.strip()
    if status:
        raise SystemExit("refusing to run the demo: these files differ from the last commit "
                         f"and the restore step would discard them:\n{status}")
    if _run(settings, "dvc", "status", "--quiet") != 0:
        raise SystemExit("refusing to run the demo: `dvc status` is not clean — run "
                         "`dvc repro` (and commit) first.")


def _restore(settings: Settings, *targets: str) -> None:
    _banner("restore the committed state (Git + DVC cache, no retraining)")
    subprocess.run(["git", "checkout", "--", *GUARDED], cwd=settings.project_root, check=True)
    print("$ git checkout -- " + " ".join(GUARDED))
    _run(settings, "dvc", "checkout", "--force", *targets)


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def demo_data_gate(settings: Settings) -> int:
    _require_clean(settings)
    data = settings.data_path
    original = _md5(data)

    _banner("plant three faults in data/telco_churn.csv")
    frame = pd.read_csv(data, dtype={"TotalCharges": str}, keep_default_na=False)
    frame.loc[0, "tenure"] = -3                     # impossible: negative tenure
    frame.loc[1, "MonthlyCharges"] = 5695.0         # cents written as dollars
    frame.loc[2, "InternetService"] = "No"          # but its add-ons say otherwise
    frame.to_csv(data, index=False, lineterminator="\n")
    print("row 0: tenure -3 | row 1: MonthlyCharges 5695.0 | "
          "row 2: InternetService 'No' with internet add-ons")

    _banner("dvc repro must stop at validate; prepare must not run")
    code = _run(settings, "dvc", "repro")
    report = json.loads(settings.validation_path.read_text(encoding="utf-8"))
    from .validation import format_validation

    print(format_validation(report))

    _restore(settings, "data/telco_churn.csv.dvc")
    restored = _md5(data)
    print(f"data md5 before {original} / after restore {restored}")
    _run(settings, "dvc", "status")

    refused = code != 0 and not report["valid"]
    print("\nRESULT:", "the data gate REFUSED, as it should" if refused
          else "the data gate did NOT refuse (bug)")
    return 0 if refused and restored == original else 1


def demo_model_gate(settings: Settings) -> int:
    _require_clean(settings)
    params_text = settings.params_path.read_text(encoding="utf-8")

    _banner("make the model worse: train.class_weight balanced -> none")
    worse, count = re.subn(r"(?m)^(  class_weight:) balanced\b", r"\1 none", params_text)
    if count != 1:
        raise SystemExit("expected exactly one `class_weight: balanced` line in params.yaml")
    with open(settings.params_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(worse)
    print("params.yaml: train.class_weight: none  (misses churners: recall drops)")

    _banner("dvc repro must re-train, then stop at the gate")
    code = _run(settings, "dvc", "repro")
    from .gates import format_gate

    report = json.loads(settings.gate_path.read_text(encoding="utf-8"))
    print(format_gate(report))

    _banner("publishing must refuse")
    register_code = _run(settings, "python", "src/main.py", "register")

    _banner("the model artifact still exists (it is the evidence)")
    model = settings.model_path
    print(f"{model.relative_to(settings.project_root)}: exists={model.exists()}, "
          f"{model.stat().st_size} bytes, md5 {_md5(model)}")

    _restore(settings, "train")
    _run(settings, "dvc", "status")
    restored = json.loads(settings.gate_path.read_text(encoding="utf-8"))
    print(f"restored verdict: {restored['verdict']} (criteria {restored['criteria_version']})")

    refused = code != 0 and report["verdict"] == "NO-GO" and register_code != 0
    print("\nRESULT:", "the model gate and the publish step REFUSED, as they should"
          if refused else "something did NOT refuse (bug)")
    return 0 if refused and model.exists() and restored["verdict"] == "GO" else 1
