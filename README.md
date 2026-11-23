# Telco Customer Churn — MLOps project

Predicts whether a telecom customer **churns** (leaves the company within the last month) from
their demographics, account and services: binary classification of `Churn` on the IBM Telco
Customer Churn dataset (7,043 customers, 21 columns). See [docs/proposal.md](docs/proposal.md)
and [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

This repository grows across HW1–HW5 ([HOMEWORK.md](HOMEWORK.md)). HW1 evidence:
[docs/hw1.md](docs/hw1.md).

## What is where

| Path | What |
| --- | --- |
| `compose.yaml`, `mlflow.Dockerfile` | Postgres + Silo + MLflow |
| `.env.example` | configuration template (copy to `.env`, which is git-ignored) |
| `data/telco_churn.csv.dvc` | DVC pointer to the dataset (the CSV itself is not in Git) |
| `dvc.yaml`, `dvc.lock`, `params.yaml` | the pipeline `prepare -> train -> evaluate`, its lock, its hyperparameters |
| `metrics/metrics.json` | committed test metrics of the current pipeline model |
| `models/run.json` | the MLflow run id of the current pipeline model (committed) |
| `src/mlops_project/` | the code; `src/main.py` is the CLI entry point |
| `tests/` | unit tests (no dataset and no Docker needed) |

## Host ports

This project uses the **56xx** block, so it can run next to the course labs (which use 55xx).

| Service | Host port | Container port |
| --- | --- | --- |
| MLflow UI / API | `5600` → http://127.0.0.1:5600 | 5000 |
| Silo S3 API (DVC remote) | `5610` | 9000 |
| Silo console | `5611` → http://127.0.0.1:5611 | 9001 |
| Postgres | `5632` | 5432 |

To change a port, edit it in `.env`. If you change `S3_API_PORT`, also run
`dvc remote modify --local storage endpointurl http://127.0.0.1:<port>`.

## Install, configure, run

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker with Compose, Git. Commands are for
bash (Git Bash on Windows); PowerShell equivalents are below.

```bash
# 1. install (Python and all dependencies, pinned by uv.lock)
uv sync --all-groups
uv run pytest -q                       # passes with Docker stopped

# 2. configure: the placeholder values work as they are for a local stack
cp .env.example .env
set -a && . ./.env && set +a           # export AWS_* so DVC can reach Silo

# 3. services
docker compose up -d --wait            # postgres, s3 (Silo), mlflow all healthy

# 4. data + pipeline
uv run dvc pull                        # fetch the dataset (and cached outputs) from Silo
uv run dvc repro                       # prepare -> train -> evaluate (up to date after a pull)
uv run dvc metrics show
uv run dvc dag
```

`dvc` needs the Silo credentials in its environment, which is why step 2 exports `.env`. The
committed `.dvc/config` holds only the bucket URL and endpoint, never a key. If `dvc` is
installed globally, plain `dvc pull` works the same as `uv run dvc pull`.

PowerShell instead of step 2:

```powershell
Copy-Item .env.example .env
Get-Content .env | Where-Object { $_ -match '^\s*[^#\s].*=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$($k.Trim())" $v.Trim() }
```

### Other commands

```bash
uv run python src/main.py sweep            # log one MLflow run per hyperparameter setting
uv run python src/main.py register         # register the pipeline run's model, alias @staging
uv run python src/main.py chain            # alias -> version -> run -> commit -> data md5 -> URI
uv run python src/main.py runs-for-data    # runs trained on the current data (--md5 for another)
uv run python src/main.py make-data --version 1   # rebuild dataset v1 (or 2) from the IBM file
docker compose down                        # stop (add -v to also delete MLflow + Silo data)
```

Each pipeline stage also runs by hand: `uv run python src/main.py prepare|train|evaluate`.

## Configuration

- **`.env`** (git-ignored): where things are, i.e. credentials, ports, the MLflow URL and the
  experiment, model and alias names.
- **`params.yaml`** (committed, hashed by DVC): what the model is, i.e. split, seed, model
  family and hyperparameters.

Changing a value in `params.yaml` makes exactly the stages that read it stale.
