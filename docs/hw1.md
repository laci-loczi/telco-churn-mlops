# HW1: Data versioning and experiment tracking (evidence)

All values below are real outputs from this repository. Commands are run from the repo root
in Git Bash, after `set -a && . ./.env && set +a` (see the [README](../README.md)).

**Short version (R9):**
`telco-churn-classifier@staging` → version **2** → run `7605f686037944c1ab6c8df07dd85e2b` →
commit `36f273d` → data md5 `3b0bfab28a8101b4e4fdd08025a5c235` →
`s3://dvc-storage/dvcstore/files/md5/3b/0bfab28a8101b4e4fdd08025a5c235`.

---

## R3: DVC tracking the dataset

The repo was initialised with a plain `dvc init` (it has its own root). `.dvc/config` is
committed. The dataset `data/telco_churn.csv` is tracked with `dvc add`: `data/.gitignore`
ignores the CSV, and the pointer `data/telco_churn.csv.dvc` is committed.

Current pointer (dataset **v2**):

```yaml
outs:
- md5: 3b0bfab28a8101b4e4fdd08025a5c235
  size: 970457
  hash: md5
  path: telco_churn.csv
```

| Version | Pointer md5 | Size (bytes) | Rows × columns (excl. header) | Commit |
| --- | --- | --- | --- | --- |
| v1 | `537887dfef07762153d506a52efb847f` | 690,380 | 5,000 × 21 | `9aa47bf` |
| v2 (current) | `3b0bfab28a8101b4e4fdd08025a5c235` | 970,457 | 7,043 × 21 | `a14c263` |

## R4: Silo as the DVC remote, both ways

`.dvc/config` (committed). It has no credentials, and `dvc remote modify --local` was never
used. `.dvc/config.local` is in `.gitignore` and is not in the repository.

```ini
[core]
    autostage = true
    analytics = false
    remote = storage
['remote "storage"']
    url = s3://dvc-storage/dvcstore
    endpointurl = http://127.0.0.1:5610
```

DVC reads the Silo keys from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, which come from
`.env`. `tests/test_versioning.py` asserts that `.dvc/config` contains no key.

**`dvc push`** uploaded both dataset versions and the pipeline outputs. The Silo listing below
was taken inside the `s3` container; the console at http://127.0.0.1:5611 shows the same
objects under `dvc-storage/dvcstore/files/md5/`. Both datasets appear as
`files/md5/<first 2>/<remaining 30>`:

```text
$ docker compose exec s3 mc ls --recursive l/dvc-storage/dvcstore/files/md5/
[2026-11-23 18:37:32 UTC] 649KiB STANDARD 13/1d2d138b5d72c611960f4dab93eb92
[2026-11-23 18:37:32 UTC] 948KiB STANDARD 3b/0bfab28a8101b4e4fdd08025a5c235   <- dataset v2
[2026-11-23 18:38:18 UTC] 7.5KiB STANDARD 4b/26c695c4fc23a13b21a95f695101a2
[2026-11-23 18:36:55 UTC] 674KiB STANDARD 53/7887dfef07762153d506a52efb847f   <- dataset v1
[2026-11-23 18:37:32 UTC] 216KiB STANDARD 73/b9f48d1673f358aae0c1587230cb66
[2026-11-23 18:37:32 UTC] 7.5KiB STANDARD a2/6779d4450df39a66afa6ccae9cf8a1
[2026-11-23 18:36:55 UTC] 7.5KiB STANDARD a3/51f8be12813239b75e10d38ba03ada
[2026-11-23 18:36:55 UTC] 462KiB STANDARD af/c645d292ea58bcc138fa1bf0bcc4e0
[2026-11-23 18:36:55 UTC] 154KiB STANDARD ea/4a9e68a826b7c4871911a934c78e82
```

(The other objects are `prepare`'s train/test splits and `train`'s `model.joblib` for each
run.)

**`dvc pull` from a fresh clone.** The repository was cloned into a new, empty directory. The
clone has no data, only the pointer. `dvc pull` fetched it with the same md5. With the Docker
stack **stopped**, the tests still pass.

```text
$ git clone <repo> fresh-clone && cd fresh-clone
$ uv sync --all-groups
$ uv run pytest -q   # Docker stack stopped
.........................s                                               [100%]
25 passed, 1 skipped in 6.96s
$ docker compose up -d --wait        # (stack started again)
$ cp .env.example .env && set -a && . ./.env && set +a
$ ls data/
README.md
processed
telco_churn.csv.dvc
$ uv run dvc pull
A       data\telco_churn.csv
A       models\model.joblib
4 files fetched and 4 files added
$ md5sum data/telco_churn.csv && grep md5 data/telco_churn.csv.dvc
3b0bfab28a8101b4e4fdd08025a5c235 *data/telco_churn.csv
- md5: 3b0bfab28a8101b4e4fdd08025a5c235
$ uv run dvc repro
'data\telco_churn.csv.dvc' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
$ uv run dvc metrics show
Path                  accuracy    f1      precision    recall    roc_auc
metrics\metrics.json  0.7501      0.6284  0.5188       0.7966    0.8461
$ git status --short
$
```

The skipped test is the live MLflow check, which skips itself when the stack is down.

## R5: Two dataset versions

The IBM file is a static snapshot, so change over time is simulated with batches (the course's
convention). `src/mlops_project/datasets.py` writes the rows byte-for-byte from the source, so
a version always has the same md5:

- **v1** (`9aa47bf`): the first 5,000 customers, the initial customer base. 26.3% churn and 8
  blank `TotalCharges`.
- **v2** (`a14c263`): v1 plus the remaining **2,043 customers as a "new signups" batch**, 7,043
  in total. The new batch churns slightly more (27.2%), and the full set has 26.5% churn and 11
  blank `TotalCharges`.

The point of v2: a model must be retrained when new customers arrive, and its runs must say
which customer base they learned from. R8 shows the runs separating by md5.

**Moving between them** (`git checkout <commit> -- <pointer>` + `dvc checkout`):

```text
$ md5sum data/telco_churn.csv   # HEAD: dataset v2
3b0bfab28a8101b4e4fdd08025a5c235 *data/telco_churn.csv
$ git checkout 9aa47bf -- data/telco_churn.csv.dvc   # the v1 commit
$ dvc checkout data/telco_churn.csv.dvc
M       data\telco_churn.csv
$ md5sum data/telco_churn.csv && wc -l data/telco_churn.csv
537887dfef07762153d506a52efb847f *data/telco_churn.csv
5001 data/telco_churn.csv
$ git checkout HEAD -- data/telco_churn.csv.dvc && dvc checkout data/telco_churn.csv.dvc
M       data\telco_churn.csv
$ md5sum data/telco_churn.csv && wc -l data/telco_churn.csv
3b0bfab28a8101b4e4fdd08025a5c235 *data/telco_churn.csv
7044 data/telco_churn.csv
```

(`wc -l` counts the header line too.)

## R6: The declared pipeline

`dvc.yaml` declares `prepare → train → evaluate`, and every `cmd` also runs by hand:

| Stage | cmd | deps (besides code) | params | outs |
| --- | --- | --- | --- | --- |
| prepare | `uv run python src/main.py prepare` | `data/telco_churn.csv` | `random_seed`, `prepare.test_size` | `data/processed/{train,test}.csv` |
| train | `uv run python src/main.py train` | `data/processed/train.csv` | `random_seed`, `train` | `models/model.joblib`, `models/run.json` (cache: false) |
| evaluate | `uv run python src/main.py evaluate` | `test.csv`, `model.joblib`, `run.json` | — | `metrics/metrics.json` (metric, cache: false) |

- The source files each stage imports (`data.py`, `model.py`, `pipeline.py`, …) are listed in
  its `deps`, so editing the training code makes `train` stale.
- Hyperparameters live in `params.yaml` and the service settings in `.env`.
- `dvc.lock` and `metrics/metrics.json` are committed.

```text
$ uv run dvc dag
+--------------------------+
| data\telco_churn.csv.dvc |
+--------------------------+
              *
              *
              *
         +---------+
         | prepare |
         +---------+
         **        **
       **            *
      *               **
+-------+               *
| train |             **
+-------+            *
         **        **
           **    **
             *  *
        +----------+
        | evaluate |
        +----------+
```

**`dvc repro` twice:** the second run re-runs nothing.

```text
$ uv run dvc repro
'data\telco_churn.csv.dvc' didn't change, skipping
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

**Changing one parameter re-runs only the affected stages.** This is commit `f17f2c4`, the sweep
winner from R7. `train.class_weight` is read only by `train`, so `prepare` is skipped:

```text
$ git diff params.yaml
-  class_weight: none      # none | balanced
+  class_weight: balanced  # none | balanced
$ uv run dvc status
train:
	changed deps:
		params.yaml:
			modified:           train
$ uv run dvc repro
'data\telco_churn.csv.dvc' didn't change, skipping
Stage 'prepare' didn't change, skipping
Running stage 'train':
trained logreg; MLflow run 81b8aea3949949a7b48936eeafc5f0fb
Running stage 'evaluate':
  "roc_auc": 0.8461,
  "f1": 0.6284,
  ...
```

## R7: MLflow tracking and the model registry

All runs land in the experiment **`telco-churn`** on the containerised server
(http://127.0.0.1:5600). Each run logs:

- its parameters
- metrics on the held-out test set: ROC-AUC, F1, precision, recall, accuracy
- the model, via `mlflow.sklearn.log_model(model, name="model")`
- tags: `git_commit`, `git_dirty`, `dvc_md5`, `dvc_remote`, `data_remote_uri`, `run_type`

**Comparison.** `uv run python src/main.py sweep` ran six settings on dataset v2. All use the
same stratified 75/25 split (seed 42, 1,761 test customers), at commit `17d9519` with a clean
tree:

| run_id | model | C | class_weight | n_estimators | max_depth | ROC-AUC | F1 | precision | recall | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a17a8c73…` | logreg | 0.1 | none | | | 0.8457 | 0.5899 | 0.6536 | 0.5375 | 0.8018 |
| `2fb9ddb0…` | logreg | 1.0 | none | | | **0.8467** | 0.6054 | 0.6633 | 0.5567 | **0.8075** |
| `1ed31cf1…` | logreg | 1.0 | **balanced** | | | 0.8461 | **0.6284** | 0.5188 | **0.7966** | 0.7501 |
| `17ee99dc…` | logreg | 10.0 | none | | | 0.8460 | 0.6054 | 0.6633 | 0.5567 | 0.8075 |
| `2a76e940…` | random_forest | | none | 200 | 8 | 0.8442 | 0.5618 | **0.6737** | 0.4818 | 0.8007 |
| `b5eb1545…` | random_forest | | none | 300 | none | 0.8236 | 0.5398 | 0.6171 | 0.4797 | 0.7831 |

**Which is best, and on which metric.**
- The proposal's primary metric is **ROC-AUC**. On it the four logistic regressions are a tie:
  0.8457–0.8467, a spread of 0.001, which is noise for a single split of 1,761 customers.
- The random forests are worse, and the unrestricted one (`max_depth=none`) overfits (0.824).
- With ROC-AUC tied, the proposal's secondary metric **F1** decides. `class_weight=balanced`
  gives the best F1 (0.628) because it raises **recall from 0.56 to 0.80**. The model then
  catches 80% of churners instead of 56%, at the cost of precision (0.52) and accuracy (0.75).
- For churn this is the right trade. A missed churner is a lost customer, while a false alarm
  costs one retention offer. Accuracy is not a useful criterion here: always predicting "stays"
  already scores 73.5%.

**Chosen:** logistic regression, C = 1.0, `class_weight=balanced`. It was set in `params.yaml`
(commit `f17f2c4`) and produced by the pipeline, so the registered model is the one that
`dvc repro` builds and `metrics/metrics.json` describes.

**Registry.** `uv run python src/main.py register` registers `runs:/<run_id>/model` from
`models/run.json` and points the alias at it:

```text
$ uv run python src/main.py register
Registered model 'telco-churn-classifier' already exists. Creating a new version of this model...
Created version '2' of model 'telco-churn-classifier'.
run 7605f686037944c1ab6c8df07dd85e2b registered as telco-churn-classifier version 2
alias @staging -> version 2
```

The registered version's `run_id` is populated: version **2** → run
**`7605f686037944c1ab6c8df07dd85e2b`**. Version 1 holds the same model from run `81b8aea3…`,
which was registered before the `git_dirty` fix (see the reflection). The alias was moved to
version 2.

Every run carries a `git_commit` tag (see the R8 table).

## R8: The data version on every run

`train` and `sweep` tag every run with `dvc_md5`, read from `data/telco_churn.csv.dvc`. They
also tag `dvc_remote` (`s3://dvc-storage/dvcstore`) and `data_remote_uri`, the object's exact
location. `uv run python src/main.py runs-for-data [--md5 <md5>]` lists the runs trained on a
data version, using `mlflow.search_runs(filter_string="tags.dvc_md5 = '<md5>'")`:

```text
$ uv run python src/main.py runs-for-data --md5 537887dfef07762153d506a52efb847f
runs with tags.dvc_md5 = '537887dfef07762153d506a52efb847f':
                          run_id tags.mlflow.runName tags.run_type params.family  metrics.roc_auc tags.git_commit
b7403228ae44423a9d97370019af151c     logreg-pipeline      pipeline        logreg            0.854        9aa47bf9

$ uv run python src/main.py runs-for-data          # default: the current pointer (v2)
runs with tags.dvc_md5 = '3b0bfab28a8101b4e4fdd08025a5c235':
                          run_id tags.mlflow.runName tags.run_type params.family  metrics.roc_auc tags.git_commit
98e2042fb88d49e2bd1cae9ffd674a52     logreg-pipeline      pipeline        logreg           0.8467        a14c2633
a17a8c73a1774ad096806e26a7bd88e6        logreg-sweep         sweep        logreg           0.8457        17d95192
2fb9ddb067554ffcbccb266f13af052a        logreg-sweep         sweep        logreg           0.8467        17d95192
1ed31cf1f28240169e5fb56c222fb0be        logreg-sweep         sweep        logreg           0.8461        17d95192
17ee99dc80234014acfde587d83ae458        logreg-sweep         sweep        logreg           0.8460        17d95192
2a76e94039d44418bb21cddfbcfab010 random_forest-sweep         sweep random_forest           0.8442        17d95192
b5eb15453e64444a8abd962b7ae7b09a random_forest-sweep         sweep random_forest           0.8236        17d95192
81b8aea3949949a7b48936eeafc5f0fb     logreg-pipeline      pipeline        logreg           0.8461        f17f2c44
```

(The v2 query was taken before the final pipeline run `7605f686…` was logged; it now appears
at the end of this list as well.)

## R9: The traceability chain

`uv run python src/main.py chain` walks it in one command:

```text
alias          : telco-churn-classifier@staging
model version  : 2
run id         : 7605f686037944c1ab6c8df07dd85e2b (logreg-pipeline)
params         : {"C": "1.0", "class_weight": "balanced", "family": "logreg", "max_iter": "1000", "random_seed": "42"}
metrics        : {"accuracy": 0.7501, "f1": 0.6284, "precision": 0.5188, "recall": 0.7966, "roc_auc": 0.8461}
git commit     : 36f273d6f4fc66d0727f2d7fdda182361f245633 (dirty=false)
data md5       : 3b0bfab28a8101b4e4fdd08025a5c235
remote URI     : s3://dvc-storage/dvcstore/files/md5/3b/0bfab28a8101b4e4fdd08025a5c235
```

Each arrow, and how to follow it yourself:

1. **alias → model version.** `telco-churn-classifier@staging` → **version 2**. In the MLflow
   UI: Models → `telco-churn-classifier` → the `@staging` alias. In code:
   `MlflowClient().get_model_version_by_alias("telco-churn-classifier", "staging")`.
2. **model version → run id.** Version 2's `run_id` is
   **`7605f686037944c1ab6c8df07dd85e2b`** (the "Source run" link on the version page).
3. **run id → params + metrics.** The run page (experiment `telco-churn`) shows the params
   `family=logreg, C=1.0, class_weight=balanced, max_iter=1000, random_seed=42` and the metrics
   `roc_auc=0.8461, f1=0.6284, precision=0.5188, recall=0.7966, accuracy=0.7501`. The same
   metrics are committed in `metrics/metrics.json`.
4. **run → git commit.** Tag `git_commit` = **`36f273d6f4fc66d0727f2d7fdda182361f245633`**, with
   `git_dirty=false`: code and params were exactly that commit. Check with
   `git show 36f273d:params.yaml`; `class_weight: balanced` is there.
5. **git commit → data md5.** Run `git show 36f273d:data/telco_churn.csv.dvc`; it gives
   md5 **`3b0bfab28a8101b4e4fdd08025a5c235`**, the same as the run's `dvc_md5` tag.
6. **data md5 → remote URI.** **`s3://dvc-storage/dvcstore/files/md5/3b/0bfab28a8101b4e4fdd08025a5c235`**
   (the run's `data_remote_uri` tag). It is visible in the Silo console at
   http://127.0.0.1:5611 under `dvc-storage/dvcstore/files/md5/3b/`. To get the bytes:
   `git checkout 36f273d -- data/telco_churn.csv.dvc && dvc pull`.

## R10: Reflection (the hardest thing)

The hardest thing was a lineage tag that was wrong while looking fine.
- **The symptom.** The first registered version (v1, run `81b8aea3…`) carried
  `git_dirty=true`, although I had committed `params.yaml` right before `dvc repro` and
  `git status` was clean before and after.
- **Diagnosis.** I listed `git_dirty` for every run and found a clean pattern. The sweep runs
  (started outside DVC) and the very first pipeline run were clean. Every *later* pipeline run
  was dirty. The difference was `dvc.lock`: it did not exist before the first run, and in every
  later run it was tracked. DVC rewrites `dvc.lock` while `dvc repro` runs, before `train`
  calls `git status`. So the tag reported the pipeline's own bookkeeping as uncommitted code.
- **The fix.** `git_dirty` now excludes the files DVC itself writes (`dvc.lock`, `metrics/`,
  `models/`), and a test covers exactly that case (`tests/test_git_dirty.py`). I also made
  `versioning.py` a `train` dependency, since it computes the tags. After rerunning, the new run
  is `dirty=false`, and I registered it as version 2 and moved `@staging` to it.
- **The lesson.** A wrong lineage tag is worse than a missing one: anyone following the chain
  would distrust a perfectly reproducible model.

A smaller trap of the same kind came from Windows line endings. With `core.autocrlf=true`, the
committed `cache: false` outputs (`metrics.json`, `run.json`) would be checked out with CRLF.
That changes their md5 and makes `dvc repro` re-run stages on a fresh clone.
`.gitattributes` forces LF, and the files are written with `\n` explicitly.
