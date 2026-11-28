"""Command-line entry point: `uv run python src/main.py <command>`.

Pipeline stages (what dvc.yaml runs; each also works by hand):
    validate        ingestion contract -> reports/validation.json (exit 1 on failure)
    prepare         model-input contract, clean + split -> data/processed/{train,test}.csv
    train           fit -> models/model.joblib, log an MLflow run -> models/run.json
    evaluate        metrics/metrics.json + metrics/slices.json, logged onto the same run
    gate            acceptance criteria -> reports/gate.json (exit 1 on NO-GO)

Everything else:
    make-data --version {1,2}   build a dataset version from the IBM download
    sweep                       log one MLflow run per hyperparameter setting
    register                    register the model + set the alias (only on a GO)
    baseline                    refresh metrics/baseline.json from the @alias champion
    measure                     the numbers the thresholds are derived from
    demo-data-gate              plant bad rows, show dvc repro refusing, restore
    demo-model-gate             make the model worse, show gate + register refusing, restore
    runs-for-data [--md5 MD5]   list runs trained on a data version (default: current)
    chain                       walk alias -> version -> run -> commit -> data md5
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings


def cmd_validate(settings, args) -> None:
    from .pipeline import GateRefused, validate
    from .validation import format_validation

    try:
        report = validate(settings)
    except GateRefused:
        print(format_validation(json.loads(settings.validation_path.read_text(encoding="utf-8"))))
        raise
    print(format_validation(report))


def cmd_prepare(settings, args) -> None:
    from .pipeline import prepare

    print(json.dumps(prepare(settings), indent=2))


def cmd_gate(settings, args) -> None:
    from .gates import format_gate
    from .pipeline import GateRefused, gate

    report = gate(settings)
    print(format_gate(report))
    if report["verdict"] != "GO":
        raise GateRefused("model gate: NO-GO")


def cmd_baseline(settings, args) -> None:
    from .registry import refresh_baseline

    baseline = refresh_baseline(settings)
    src = baseline["source"]
    print(f"metrics/baseline.json <- {src['model']}@{src['alias']} "
          f"(version {src['version']}, run {src['run_id']})")
    print(json.dumps(baseline["metrics"], indent=2))


def cmd_measure(settings, args) -> None:
    import joblib
    import pandas as pd

    from .data import features_and_target
    from .measure import bootstrap, trivial_predictors

    test = pd.read_csv(settings.processed_dir / "test.csv", dtype={"SeniorCitizen": str})
    x, y = features_and_target(test)
    model = joblib.load(settings.model_path)
    print(f"test set: {len(y)} rows, {int(y.sum())} churners ({y.mean():.4f})")
    print("trivial predictors:", json.dumps(trivial_predictors(y), indent=2))
    print("bootstrap (1000 resamples) of the current model:",
          json.dumps(bootstrap(y, model.predict(x), model.predict_proba(x)[:, 1]), indent=2))


def cmd_demo_data_gate(settings, args) -> None:
    from .demos import demo_data_gate

    sys.exit(demo_data_gate(settings))


def cmd_demo_model_gate(settings, args) -> None:
    from .demos import demo_model_gate

    sys.exit(demo_model_gate(settings))


def cmd_train(settings, args) -> None:
    from .pipeline import train

    result = train(settings)
    if result["run_id"]:
        print(f"trained {result['family']}; MLflow run {result['run_id']}")
    else:
        print(f"trained {result['family']}; MLflow not reachable, no run logged")


def cmd_evaluate(settings, args) -> None:
    from .pipeline import evaluate

    print(json.dumps(evaluate(settings)["metrics"], indent=2))


def cmd_make_data(settings, args) -> None:
    from .datasets import build_version, download_source

    source = download_source(settings.source_path)
    rows = build_version(args.version, source, settings.data_path)
    print(f"built dataset version {args.version}: {rows} rows -> data/telco_churn.csv")
    print("next: `dvc add data/telco_churn.csv` to record this version")


def cmd_sweep(settings, args) -> None:
    from .pipeline import sweep

    results = sweep(settings)
    header = ["run_id", "family", "C", "class_weight", "n_estimators", "max_depth",
              "roc_auc", "f1", "precision", "recall", "accuracy"]
    print(" | ".join(header))
    for row in results:
        print(" | ".join(str(row.get(key, "-")) for key in header))


def cmd_register(settings, args) -> None:
    from .registry import register

    result = register(settings)
    state = "already registered as" if result["already_registered"] else "registered as"
    print(result["reason"])
    print(f"run {result['run_id']} {state} {result['name']} version {result['version']}")
    print(f"alias @{result['alias']} -> version {result['version']}")


def cmd_runs_for_data(settings, args) -> None:
    from .tracking import mlflow_available, search_runs_by_data_version
    from .versioning import pointer_md5

    md5 = args.md5 or pointer_md5(settings.pointer_path)
    if not mlflow_available(settings):
        raise SystemExit(f"MLflow not reachable at {settings.mlflow_tracking_uri}")
    print(f"runs with tags.dvc_md5 = '{md5}':")
    frame = search_runs_by_data_version(settings, md5)
    if frame.empty:
        print("  (none)")
        return
    columns = [c for c in ["run_id", "tags.mlflow.runName", "tags.run_type", "params.family",
                           "metrics.roc_auc", "tags.git_commit"] if c in frame.columns]
    out = frame[columns].copy()
    if "tags.git_commit" in out.columns:
        out["tags.git_commit"] = out["tags.git_commit"].str[:8]
    print(out.to_string(index=False))


def cmd_chain(settings, args) -> None:
    from .registry import chain

    c = chain(settings)
    print(f"alias          : {c['alias']}")
    print(f"model version  : {c['version']}")
    print(f"run id         : {c['run_id']} ({c['run_name']})")
    print(f"params         : {json.dumps(c['params'], sort_keys=True)}")
    print(f"metrics        : {json.dumps(c['metrics'], sort_keys=True)}")
    print(f"git commit     : {c['git_commit']} (dirty={c['git_dirty']})")
    print(f"data md5       : {c['dvc_md5']}")
    print(f"remote URI     : {c['data_remote_uri']}")


COMMANDS = {
    "validate": cmd_validate,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "gate": cmd_gate,
    "make-data": cmd_make_data,
    "sweep": cmd_sweep,
    "register": cmd_register,
    "baseline": cmd_baseline,
    "measure": cmd_measure,
    "demo-data-gate": cmd_demo_data_gate,
    "demo-model-gate": cmd_demo_model_gate,
    "runs-for-data": cmd_runs_for_data,
    "chain": cmd_chain,
}


def _utf8_console() -> None:
    """MLflow prints emoji; a default Windows console (cp1252) crashes on them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> None:
    _utf8_console()
    parser = argparse.ArgumentParser(prog="mlops-project", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--version", type=int, default=2, help="make-data: dataset version")
    parser.add_argument("--md5", help="runs-for-data: the data md5 (default: current pointer)")
    args = parser.parse_args(argv)
    from .pipeline import GateRefused
    from .registry import PublishRefused

    try:
        COMMANDS[args.command](load_settings(), args)
    except (GateRefused, PublishRefused) as refusal:
        # A refusal is an answer, not a crash: one clear line and a non-zero exit,
        # which is what makes `dvc repro` stop. Flush stdout first so the report
        # prints before the refusal line.
        sys.stdout.flush()
        print(f"REFUSED: {refusal}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
