"""Command-line entry point: `uv run python src/main.py <command>`.

Pipeline stages (what dvc.yaml runs; each also works by hand):
    prepare         clean + stratified split -> data/processed/{train,test}.csv
    train           fit -> models/model.joblib, log an MLflow run -> models/run.json
    evaluate        test metrics -> metrics/metrics.json, logged onto the same run

Everything else:
    make-data --version {1,2}   build a dataset version from the IBM download
    sweep                       log one MLflow run per hyperparameter setting
    register                    register the pipeline run's model + set the alias
    runs-for-data [--md5 MD5]   list runs trained on a data version (default: current)
    chain                       walk alias -> version -> run -> commit -> data md5
"""

from __future__ import annotations

import argparse
import json

from .config import load_settings


def cmd_prepare(settings, args) -> None:
    from .pipeline import prepare

    print(json.dumps(prepare(settings), indent=2))


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
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "make-data": cmd_make_data,
    "sweep": cmd_sweep,
    "register": cmd_register,
    "runs-for-data": cmd_runs_for_data,
    "chain": cmd_chain,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlops-project", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--version", type=int, default=2, help="make-data: dataset version")
    parser.add_argument("--md5", help="runs-for-data: the data md5 (default: current pointer)")
    args = parser.parse_args(argv)
    COMMANDS[args.command](load_settings(), args)


if __name__ == "__main__":
    main()
