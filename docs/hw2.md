# HW2: Data validation and evaluation gates

The pipeline is now `validate → prepare → train → evaluate → gate`. It can say no at two
points, for reasons written down before the run:

- **Entrance:** `validate` runs the ingestion contract. `prepare` depends on its verdict
  (`reports/validation.json`), so bad data stops the pipeline before any work is done.
- **Exit:** `gate` checks the model against the `acceptance` block in `params.yaml`
  (`criteria_version: "1.0.0"`) and exits non-zero on a NO-GO. `register` refuses to publish
  unless `reports/gate.json` is a GO for this exact run.

The committed state is a **GO: 14 passed, 0 failed, 1 skipped**.

All numbers below come from this repository. Re-measure them with
`uv run python src/main.py measure` (trivial predictors and bootstrap) and
`cat metrics/slices.json`. The test set has 1,761 customers, 467 of them churners (26.52%).

---

## R1: The two contracts (`src/mlops_project/schemas.py`)

**`IngestionSchema`** describes the CSV as it arrives. It has `strict = True` and
`coerce = False`, declares the dtype of all 21 columns, and gives every numeric column a range:

| Column | Rule | Measured in the real data | Why this bound |
| --- | --- | --- | --- |
| `tenure` | int, 0–72 | min 0, max 72 | The IBM extract covers 6 years; 73+ means a different extract or a corrupt value |
| `MonthlyCharges` | float, 15–150 | min 18.25, max 118.75 | The cheapest plan is 18.25 and the full bundle 118.75. The headroom allows a price change, but rejects a sign error or cents written as dollars |
| `SeniorCitizen` | int, {0, 1} | 0/1 only | a flag |
| `TotalCharges` | text: blank or a non-negative number | 11 blanks, max 8,684.80 | It arrives as text (the known trap) |
| `customerID` | `NNNN-XXXXX`, unique | all match, 0 duplicates | an identifier |
| 16 categoricals | `isin` their documented levels | — | A new level means a new product, and someone has to decide how to encode it |

It also has four row-level checks, and **all four hold on all 7,043 real rows**:

- `TotalCharges` is blank **exactly** when `tenure == 0` (11 = 11).
- `TotalCharges / (tenure × MonthlyCharges)` lies within **0.5–2.0**. The real data spans
  **0.689–1.573**, because today's price differs from earlier months' prices. Outside that band
  the two columns no longer describe the same customer.
- An add-on is `No internet service` exactly when `InternetService == "No"`.
- `MultipleLines == "No phone service"` exactly when `PhoneService == "No"`.

**`ModelInputSchema`** describes what the model may see, after `clean()`. It is checked in
`prepare`. It is stricter in three ways that matter for this dataset, each with a comment in
the code:

1. **No `customerID`** (strict columns). An identifier the model can see is a leakage path.
2. **`TotalCharges` is a float and never null.** A blank is legal on arrival but not in model
   input, so the blank → 0 rule must already have run.
3. **`SeniorCitizen` is the string `"0"`/`"1"`.** As an int it would be *scaled* like a number
   while its 15 Yes/No neighbours are one-hot encoded.

Both contracts pass on the real data. Removing one cross-field rule would let rows through that
describe impossible customers; tightening `MonthlyCharges` to the observed 18.25–118.75 would
reject the first legitimate price change.

## R2: The data gate (`validate`, and the edge into `prepare`)

`validate` writes `reports/validation.json` (`cache: false`, no timestamp, failures in a fixed
order), then exits 1 if the contract failed. `prepare` lists that file in its `deps`. That edge
is the gate: on bad data, `prepare` is never reached. `prepare` also re-reads the verdict and
refuses if it is not a pass.

**Demonstration**, runnable verbatim. It plants three faults, shows `dvc repro` refusing, and
restores the committed data from the DVC cache (it refuses to start on uncommitted pipeline
changes):

```text
$ uv run python src/main.py demo-data-gate
$ dvc status --quiet
[exit code 0]
=== plant three faults in data/telco_churn.csv ===
row 0: tenure -3 | row 1: MonthlyCharges 5695.0 | row 2: InternetService 'No' with internet add-ons
=== dvc repro must stop at validate; prepare must not run ===
$ dvc repro
Verifying data sources in stage: 'data\telco_churn.csv.dvc'
Running stage 'validate':
> uv run python src/main.py validate
IngestionSchema: FAILED on 7043 rows
  FAIL <frame>          no_internet_means_no_addons              rows=1 first=[2]
  FAIL <frame>          total_matches_tenure_times_monthly       rows=1 first=[1]
  FAIL MonthlyCharges   less_than_or_equal_to(150.0)             rows=1 first=[1] values=[5695.0]
  FAIL tenure           greater_than_or_equal_to(0)              rows=1 first=[0] values=[-3]
REFUSED: data contract failed: 4 check(s)
ERROR: failed to reproduce 'validate': failed to run: uv run python src/main.py validate, exited with 1
[exit code 255]
IngestionSchema: FAILED on 7043 rows
  FAIL <frame>          no_internet_means_no_addons              rows=1 first=[2]
  FAIL <frame>          total_matches_tenure_times_monthly       rows=1 first=[1]
  FAIL MonthlyCharges   less_than_or_equal_to(150.0)             rows=1 first=[1] values=[5695.0]
  FAIL tenure           greater_than_or_equal_to(0)              rows=1 first=[0] values=[-3]
=== restore the committed state (Git + DVC cache, no retraining) ===
$ git checkout -- params.yaml dvc.lock models/run.json metrics reports data/telco_churn.csv.dvc
$ dvc checkout --force data/telco_churn.csv.dvc
M       data\telco_churn.csv
[exit code 0]
data md5 before 3b0bfab28a8101b4e4fdd08025a5c235 / after restore 3b0bfab28a8101b4e4fdd08025a5c235
$ dvc status
Data and pipelines are up to date.
[exit code 0]
RESULT: the data gate REFUSED, as it should
```

All four failures are caught, including the two row-level ones. The cents-as-dollars row
(5,695.00) breaks both the `MonthlyCharges` bound and the `TotalCharges` ratio rule. `prepare`
never ran, and the data's md5 is identical after the restore.

## R3: Slice evaluation (`metrics/slices.json`)

The slices are declared in `params.yaml` under `evaluation.slices`; none is hard-coded:

| Slice | Who is in it | Why a stakeholder cares |
| --- | --- | --- |
| `contract` | month-to-month / one-year / two-year customers | Retention offers are priced by contract, and the churn rate is 43% / 11% / 3% |
| `tenure_band` | 0–12, 13–48, 49–72 months | New customers are where churn concentrates (47% at 0–12m vs 8% at 49–72m), and the offer differs by loyalty |
| `senior_citizen` | non-senior / senior | Age is the sensitive attribute with a business stake (seniors churn 41.7% vs 23.6%); it is also the fairness axis (R8) |
| `gender` | Female / Male | A sensitive attribute, reported to show the model does not depend on it |

Every row carries `n` and the number of churners (`positives`):

| Slice | Group | n | churners | predicted churn | selection rate | precision | recall (TPR) | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contract | Month-to-month | 968 | 414 | 684 | 0.7066 | 0.5292 | 0.8744 | 0.6594 |
| contract | One year | 366 | 43 | 33 | 0.0902 | 0.3030 | 0.2326 | 0.2632 |
| contract | Two year | 427 | 10 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tenure_band | 0-12m | 534 | 251 | 360 | 0.6742 | 0.5889 | 0.8446 | 0.6939 |
| tenure_band | 13-48m | 659 | 172 | 306 | 0.4643 | 0.4608 | 0.8198 | 0.5900 |
| tenure_band | 49-72m | 568 | 44 | 51 | 0.0898 | 0.3725 | 0.4318 | 0.4000 |
| senior_citizen | non_senior | 1475 | 345 | 521 | 0.3532 | 0.4933 | 0.7449 | 0.5935 |
| senior_citizen | senior | 286 | 122 | 196 | 0.6853 | 0.5867 | 0.9426 | 0.7233 |
| gender | Female | 863 | 240 | 360 | 0.4171 | 0.5306 | 0.7958 | 0.6367 |
| gender | Male | 898 | 227 | 357 | 0.3976 | 0.5070 | 0.7974 | 0.6199 |

**What the table shows.** The aggregate F1 of 0.6284 hides a model that is good on short,
month-to-month customers and weak on long-commitment ones. It reaches F1 0.26 on one-year
contracts and 0.40 after 48 months. On two-year contracts it catches **none** of the 10
churners. That group is too small to score, and the gate reports it as skipped rather than
dropping it or scoring F1 on 10 positives. For the business this is a known weakness, and
mostly a tolerable one: retention effort goes to month-to-month customers, where 414 of the
467 test churners are. It is still written down here rather than averaged away.

**Raw values.** The slices are computed on the test split's raw columns. `clean()` changes only
three things:
- `TotalCharges`: blank → 0
- `Churn`: Yes/No → 1/0
- `SeniorCitizen`: from int to string, with the same values

No slice column is imputed. All scaling and encoding happens inside the sklearn `Pipeline`, at
prediction time.

The one imputed column is `TotalCharges`, and it affects **3 of the 1,761** test rows (`tenure =
0`). Had I sliced on a `TotalCharges` band after imputation:
- Under the current rule (blank → 0), those 3 brand-new customers land in the lowest band,
  which is where they really belong (never billed).
- Under a mean imputation they would have been given **2,283.30** (the median is 1,397.48).
  That would put the highest-risk customers, new ones (47% churn at 0–12m), into a mid-spend
  band. It would inflate that band's churn rate and make the model's error there look like a
  property of mid-spend customers.

Slicing after scaling would make the tenure bins meaningless, because they would be cut in
z-units.

## R4: Acceptance criteria and where each number comes from

```yaml
acceptance:
  criteria_version: "1.0.0"
  minimum:        { roc_auc: 0.80, f1: 0.55, recall: 0.70 }
  regression:     { primary_metric: roc_auc, max_drop: 0.01, require_baseline: true }
  slice_rules:    { min_support: 100, min_positives: 30, min_f1: 0.20, max_f1_gap: 0.40 }
  fairness:       { senior_citizen: { max_tpr_gap: 0.25 } }
```

**Measurements** (`uv run python src/main.py measure`, 1,000 bootstrap resamples of the test
set):

| | ROC-AUC | F1 | Recall | Accuracy |
| --- | --- | --- | --- | --- |
| current model (logreg, balanced) | 0.8461 | 0.6284 | 0.7966 | 0.7501 |
| bootstrap std | 0.0095 | 0.0159 | 0.0184 | |
| bootstrap 95% interval | 0.8261–0.8636 | 0.5960–0.6581 | 0.7586–0.8327 | |
| **always predicts "stays"** | 0.5 | **0.0** | 0.0 | **0.7348** |
| **always predicts "churns"** | 0.5 | **0.4192** | 1.0 | 0.2652 |

**The trivial baseline.** A constant "stays" predictor scores **accuracy 0.7348**, close to
the model's 0.7501. That is why accuracy has no floor at all: it cannot tell a working model
from a useless one on this data. A constant "churns" predictor scores **F1 0.4192** and recall
1.0, and any constant scores ROC-AUC 0.5. All the floors sit above every trivial predictor
(F1 0.55 > 0.4192, ROC-AUC 0.80 > 0.5), and the recall floor is only meaningful together with
the F1 floor, which the always-churn model fails.

**Floors:**
- **`roc_auc ≥ 0.80`:** 4.8 bootstrap std below the current 0.8461, and below the lower end
  of its 95% interval (0.8261). It is a "this is broken" line, not a "this got slightly worse"
  line. In the HW1 sweep every sensible model cleared it, including the overfit unrestricted
  forest at 0.8236; the regression check is what catches that one.
- **`f1 ≥ 0.55`:** 4.9 std below 0.6284, and 0.13 above always-churn (0.4192). From the HW1
  sweep, the depth-8 forest (0.5618) would pass and the unrestricted forest (0.5398) would
  fail.
- **`recall ≥ 0.70`:** this encodes the business decision made in HW1. A missed churner is a
  lost customer, while a false alarm costs one retention offer. It sits 5.3 std below the
  current 0.7966. The unweighted logistic regression from the HW1 sweep (recall 0.5567)
  **fails** it on purpose; that is exactly the R6 demonstration.

**Regression** (`roc_auc`, `max_drop 0.01`, against `metrics/baseline.json`):
- ROC-AUC is the proposal's primary metric, and the one the HW1 model choice started from.
- 0.01 is about **one bootstrap std** of ROC-AUC (0.0095) on this test set. A smaller drop is
  within resampling noise and not evidence of a worse model.
- Champion and challenger are scored on the same split, so the noise in their *difference* is
  smaller than that, which makes 0.01 conservative rather than tight.
- For scale: the four HW1 logistic regressions differed by at most 0.001, and the
  unrestricted forest would drop 0.0225 and fail.
- `require_baseline: true` makes a missing baseline a NO-GO (R7).

**Slice rules:**
- **`min_support 100`, `min_positives 30`.** Even at 43 churners, the one-year F1 has a
  bootstrap std of **0.068** (the 49–72m group, 44 churners: 0.066). Below about 30 positives
  F1 is mostly noise. Two-year contracts (n = 427 but **10** churners) are therefore skipped
  and reported, never gated on 10 rows.
- **`min_f1 0.20`, `max_f1_gap 0.40`.** These are set from the current worst scored group,
  one-year: F1 **0.2632**, 0.3652 below the aggregate. I am stating the inversion plainly:
  these two numbers were chosen *after* seeing the model, so they certify the current
  weakness instead of judging it. What they do is catch it getting *worse*. A group collapsing
  to near-zero F1 fails, and the demo in R6 shows exactly that (one-year F1 → 0.0). They are
  within about one std of the current value, so a retrain on new data could trip them
  through noise alone. I accept that for the weakest group: a NO-GO there is worth a look.

**Fairness:** `senior_citizen` TPR gap ≤ **0.25**, with a measured gap of **0.1977** and a
bootstrap std of **0.031**, so the margin is 1.7 std. See R8.

**The inversion, honestly.** A real project would fix these criteria before training, from
business requirements, and test the model against them. Here every threshold was chosen
after seeing this model's numbers. The floors are far enough below them (about 5 std) that
this barely matters. The slice thresholds are not, which is why their justification above
admits it.

## R5: The gate stage (`reports/gate.json`)

- `gate` reads `metrics/metrics.json`, `metrics/slices.json`, `metrics/baseline.json`, the
  `acceptance` block and `models/run.json`. It calls `gates.evaluate_gate()`, a pure function
  over dicts, writes the verdict (`cache: false`), and exits 1 on a NO-GO.
- Each check is **pass**, **fail** or **skipped**, and a skipped check carries its reason.
- The summary line, and `summary` in the JSON, show the skipped count: `GATE GO (criteria
  1.0.0): 14 passed, 0 failed, 1 SKIPPED`.
- The file has no timestamp and is written with sorted keys and LF endings. It is
  byte-identical across `dvc repro -f -s gate` reruns (checked by md5) and across the two
  `dvc repro` runs the grader does; a unit test checks the same.
- It also records the `run_id` it judged, so a verdict cannot be reused for a different model.

## R6: The gate has teeth

There is one publish path, `uv run python src/main.py register`. It calls
`registry.publish_decision()` **before any registry call**, and that refuses unless
`reports/gate.json` says GO **and** was written for the run in `models/run.json`. A stale GO
from an earlier model is refused too. The gate never deletes the model.

**Demonstration**, runnable verbatim. It changes one hyperparameter (`train.class_weight:
balanced → none`), shows `dvc repro` stopping at the gate, shows `register` refusing, shows
the model file still exists, and restores the committed state from Git and the DVC cache. No
retraining is needed for the restore.

```text
$ uv run python src/main.py demo-model-gate
$ dvc status --quiet
[exit code 0]
=== make the model worse: train.class_weight balanced -> none ===
params.yaml: train.class_weight: none  (misses churners: recall drops)
=== dvc repro must re-train, then stop at the gate ===
$ dvc repro
'data\telco_churn.csv.dvc' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'prepare' didn't change, skipping
Running stage 'train':
> uv run python src/main.py train
🏃 View run logreg-pipeline at: http://127.0.0.1:5600/#/experiments/1/runs/23a088090e3e495bb5869150530ea298
trained logreg; MLflow run 23a088090e3e495bb5869150530ea298
Updating lock file 'dvc.lock'
Running stage 'evaluate':
> uv run python src/main.py evaluate
{
  "roc_auc": 0.8467,
  "f1": 0.6054,
  "precision": 0.6633,
  "recall": 0.5567,
  "accuracy": 0.8075
}
Updating lock file 'dvc.lock'
Running stage 'gate':
> uv run python src/main.py gate
GATE NO-GO (criteria 1.0.0): 11 passed, 3 failed, 1 SKIPPED
  PASS     floor.f1                                     observed=0.6054, threshold=0.55
  FAIL     floor.recall                                 observed=0.5567, threshold=0.7
  PASS     floor.roc_auc                                observed=0.8467, threshold=0.8
  PASS     regression.roc_auc                           observed=0.8467, baseline=0.8461, delta=0.0006, threshold=0.01
  PASS     slice.contract.Month-to-month                f1=0.6452, gap=-0.0398
  FAIL     slice.contract.One year                      f1=0.0, gap=0.6054
  SKIPPED  slice.contract.Two year                      too small to score: n=427 (min 100), positives=10 (min 30)
  PASS     slice.gender.Female                          f1=0.5977, gap=0.0077
  PASS     slice.gender.Male                            f1=0.6132, gap=-0.0078
  PASS     slice.senior_citizen.non_senior              f1=0.571, gap=0.0344
  PASS     slice.senior_citizen.senior                  f1=0.6911, gap=-0.0857
  PASS     slice.tenure_band.0-12m                      f1=0.6827, gap=-0.0773
  PASS     slice.tenure_band.13-48m                     f1=0.5559, gap=0.0495
  FAIL     slice.tenure_band.49-72m                     f1=0.125, gap=0.4804
  PASS     fairness.senior_citizen.tpr_gap              observed=0.1895, threshold=0.25
  -> NO-GO because: floor.recall
  -> NO-GO because: slice.contract.One year
  -> NO-GO because: slice.tenure_band.49-72m
REFUSED: model gate: NO-GO
ERROR: failed to reproduce 'gate': failed to run: uv run python src/main.py gate, exited with 1
[exit code 255]
GATE NO-GO (criteria 1.0.0): 11 passed, 3 failed, 1 SKIPPED
  PASS     floor.f1                                     observed=0.6054, threshold=0.55
  FAIL     floor.recall                                 observed=0.5567, threshold=0.7
  PASS     floor.roc_auc                                observed=0.8467, threshold=0.8
  PASS     regression.roc_auc                           observed=0.8467, baseline=0.8461, delta=0.0006, threshold=0.01
  PASS     slice.contract.Month-to-month                f1=0.6452, gap=-0.0398
  FAIL     slice.contract.One year                      f1=0.0, gap=0.6054
  SKIPPED  slice.contract.Two year                      too small to score: n=427 (min 100), positives=10 (min 30)
  PASS     slice.gender.Female                          f1=0.5977, gap=0.0077
  PASS     slice.gender.Male                            f1=0.6132, gap=-0.0078
  PASS     slice.senior_citizen.non_senior              f1=0.571, gap=0.0344
  PASS     slice.senior_citizen.senior                  f1=0.6911, gap=-0.0857
  PASS     slice.tenure_band.0-12m                      f1=0.6827, gap=-0.0773
  PASS     slice.tenure_band.13-48m                     f1=0.5559, gap=0.0495
  FAIL     slice.tenure_band.49-72m                     f1=0.125, gap=0.4804
  PASS     fairness.senior_citizen.tpr_gap              observed=0.1895, threshold=0.25
  -> NO-GO because: floor.recall
  -> NO-GO because: slice.contract.One year
  -> NO-GO because: slice.tenure_band.49-72m
=== publishing must refuse ===
$ python src/main.py register
REFUSED: refusing to register: gate verdict is NO-GO: floor.recall; slice.contract.One year; slice.tenure_band.49-72m
[exit code 1]
=== the model artifact still exists (it is the evidence) ===
models\model.joblib: exists=True, 7681 bytes, md5 a26779d4450df39a66afa6ccae9cf8a1
=== restore the committed state (Git + DVC cache, no retraining) ===
$ git checkout -- params.yaml dvc.lock models/run.json metrics reports data/telco_churn.csv.dvc
$ dvc checkout --force train
M       models\model.joblib
[exit code 0]
$ dvc status
Data and pipelines are up to date.
[exit code 0]
restored verdict: GO (criteria 1.0.0)
RESULT: the model gate and the publish step REFUSED, as they should
```

**Why this model counts as genuinely worse even though its ROC-AUC went *up* (0.8467 vs
0.8461).** The two models rank customers almost identically. But at the decision threshold,
the unweighted model misses **44%** of churners (recall 0.5567, against 20% before). On
one-year contracts it finds **none** of them (F1 0.0), and after 48 months it drops to F1
0.125. A gate on ROC-AUC alone would have published it. The recall floor and the slice rules
are what said no.

The registry still holds versions 1 and 2 only, with `@staging` → 2. The refused model never
got a version. Its MLflow run exists, tagged `git_dirty=true`, which is right: it was trained
from an uncommitted `params.yaml`.

The committed `params.yaml` has `class_weight: balanced`, and the committed state is a GO.

## R7: The recorded baseline (`metrics/baseline.json`)

`metrics/baseline.json` holds the champion's metrics. It is committed and is a `gate`
dependency. Refresh it from the **alias**, never a version number:

```text
$ uv run python src/main.py baseline
metrics/baseline.json <- telco-churn-classifier@staging (version 2, run 7605f686037944c1ab6c8df07dd85e2b)
```

The file records the alias, the version it resolved to and the run id, with no timestamp.
Because the current pipeline model reproduces the champion exactly (same data, code and
params), today's regression delta is 0.0.

**No-baseline path**, runnable verbatim (it restores the file afterwards):

```text
$ mv metrics/baseline.json metrics/baseline.json.bak
$ uv run python src/main.py gate
GATE NO-GO (criteria 1.0.0): 13 passed, 0 failed, 2 SKIPPED
  SKIPPED  regression.roc_auc                           no baseline recorded (metrics/baseline.json missing or empty)
  SKIPPED  slice.contract.Two year                      too small to score: n=427 (min 100), positives=10 (min 30)
  -> NO-GO because: no baseline recorded, and regression.require_baseline is true
REFUSED: model gate: NO-GO
[exit code 1]
$ mv metrics/baseline.json.bak metrics/baseline.json && git checkout -- reports/gate.json
```

The regression check is **skipped** with a reason, the skipped count goes from 1 to 2, and the
verdict is **NO-GO**, not silently upgraded to a pass. To let a project's very first model
through, you would set `require_baseline: false` deliberately. The skip would still be counted
and printed (`tests/test_gates.py` covers both cases).

## R8: Fairness: one rate-based gap

The gaps are computed in `slices.json` under `fairness`, using only groups that meet the
`min_support`/`min_positives` rules:

| Axis | Selection rate | TPR (recall) | Selection-rate gap | TPR gap | Treatment |
| --- | --- | --- | --- | --- | --- |
| `senior_citizen` | non-senior 0.3532, senior 0.6853 | non-senior 0.7449, senior 0.9426 | 0.3321 | **0.1977** | **TPR gap gated ≤ 0.25**; selection gap reported |
| `gender` | Female 0.4171, Male 0.3976 | Female 0.7958, Male 0.7974 | 0.0195 | 0.0016 | reported only |

**Why age, and not gender:**
- Gender shows no gap (TPR gap 0.0016), and the data gives no reason to expect one: women and
  men churn at essentially the same rate (26.92% vs 26.16%). Gating it would add a check that
  cannot fail for a real reason.
- Age has both a real base-rate difference and a real stake. Seniors churn at 41.7% against
  23.6%, and an older customer who leaves because nobody offered a better plan is a customer
  the retention programme exists to reach.

**Why gate TPR and not the selection rate.**
- The selection-rate gap (0.33) is mostly the base-rate difference. Seniors *should* be
  flagged more often because they churn more often, so gating it would push the model to
  ignore a real signal.
- The TPR gap asks a different question: of the customers who are actually leaving, does the
  model find seniors and non-seniors at the same rate? This is equal opportunity: an offer
  should reach churners regardless of age.
- Today **non-seniors** are the under-served group (0.74 vs 0.94). I gate the gap at 0.25,
  which is 1.7 bootstrap std above the measured 0.1977.

**A group I would want to audit for and cannot: low-income customers.**
- A churn model drives *who gets a retention discount*. If it systematically misses churners
  who leave because they cannot afford the bill, the discount goes to customers who would have
  stayed anyway, while the most price-sensitive ones leave without an offer.
- The dataset has no income, no social-tariff flag, and no billing address that could be
  linked to area-level deprivation data.
- `MonthlyCharges` and `PaymentMethod` are not proxies I would trust. Electronic check is
  associated with churn, but that is not income.
- To make the audit possible, the telco would have had to **record an income band or
  social-tariff eligibility at signup**, with consent and a stated purpose (it is personal
  data under GDPR). It would also have had to **keep it linkable to the customer record in the
  extract**.
- Adding it now would not help historical customers: the label is "churned last month", so
  for the customers in this extract, that point has passed.

## What this gate cannot catch

- **Plausible-but-wrong values.** The contract checks what a row *may* be, not whether it is
  *true*. A monthly charge of 70.35 recorded as 80.35 passes every rule. So does a churn label
  flipped from Yes to No, and the gate cannot detect a wrong label at all.
- **Change over time.** The test set comes from the same static snapshot as the training set,
  split at random. Nothing here says the model will work on next month's customers. A shift
  in the contract mix would pass both gates, and drift is deliberately out of scope (Weeks
  11–12).
- **The two-year contract customers.** They are 24% of the test set, and the gate is blind to
  them: 10 churners is too few to score. It reports that loudly, but a model that never flags
  a two-year customer (like this one) passes.
- **Its own thresholds.** The slice thresholds were fitted to this model (R4). The gate proves
  the model has not got *worse* than itself, not that it is good enough for long-contract
  customers.
- **A baseline scored on a different test set.** The regression check compares numbers. If
  `prepare.test_size` or the seed changed, champion and challenger would be scored on
  different rows, and the delta would mean nothing. The gate does not check that the baseline
  was measured on the same split.
- **A weak baseline.** Today the baseline *is* this model (delta 0.0). Refreshing the baseline
  to a worse champion lowers the bar, and the gate trusts whatever `@staging` points at.
- **Tracking failures.** During one run of the R6 demonstration, `train`'s MLflow health check
  failed for a moment and the stage carried on without logging a run. The pipeline did not
  fail. `register` still refused, because there was no run id to publish, but the gap is real:
  a model can be built with no MLflow record of it.
- **The decision threshold.** All metrics use the default 0.5 probability cut-off. Calibration,
  and the cost trade-off behind the threshold, are not checked.
