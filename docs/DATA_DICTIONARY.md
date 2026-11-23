# Data Dictionary — Telco Customer Churn

## Overview

- **Dataset / source / license:** IBM "Telco Customer Churn" sample dataset, a fictional
  California telco's customers. Source:
  <https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv>
  (IBM sample data, Apache-2.0 repository; also mirrored on Kaggle as
  `blastchar/telco-customer-churn`).
- **Rows / columns:** 7,043 customers × 21 columns (1 identifier, 19 features, 1 target).
  Dataset v1 used in this project is the first 5,000 rows; v2 is all 7,043.
- **Task:** binary classification. Target column `Churn` (`Yes` → 1, `No` → 0).
- **Class balance:** 26.5% churn (1,869 of 7,043). Moderately imbalanced.

## Features

| Column | Meaning | Unit / type | Plausible range | Untrustworthy? | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customerID` | customer identifier | string `NNNN-XXXXX` | unique (0 duplicates) | — | **dropped**: no signal, leakage risk |
| `gender` | customer gender | category | `Female`, `Male` | — | sensitive attribute; check for fairness later |
| `SeniorCitizen` | senior citizen | 0/1 | `0`, `1` | — | the only binary stored as 0/1 (rest are Yes/No); encoded as a category |
| `Partner` | has a partner | Yes/No | | — | |
| `Dependents` | has dependents | Yes/No | | — | |
| `tenure` | months with the company | integer months | 0–72 | — | `0` = brand-new customer, never billed |
| `PhoneService` | has phone service | Yes/No | | — | |
| `MultipleLines` | multiple phone lines | category | `Yes`, `No`, `No phone service` | — | the third value is a real category, not missing |
| `InternetService` | internet type | category | `DSL`, `Fiber optic`, `No` | — | strong churn signal (fiber churns more) |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | add-on services | category | `Yes`, `No`, `No internet service` | — | "No internet service" is a category, not a null |
| `Contract` | contract term | category | `Month-to-month`, `One year`, `Two year` | — | the strongest single predictor |
| `PaperlessBilling` | paperless billing | Yes/No | | — | |
| `PaymentMethod` | payment method | category | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | — | |
| `MonthlyCharges` | current monthly charge | USD | 18.25–118.75 | — | |
| `TotalCharges` | lifetime charges | USD, **stored as text** | 0–8,684.80 | **yes: 11 blanks** | blank exactly when `tenure = 0`; set to 0 in `prepare` |
| `Churn` | churned last month (**target**) | Yes/No | | — | mapped to 1/0 |

## Known quirks / caveats

- **`TotalCharges` is the trap.** It looks numeric, but it loads as text because 11 rows hold a
  single space. All 11 are customers with `tenure = 0` who have never been billed, so the
  missingness has a cause. `prepare` sets those totals to **0** rather than imputing a mean, and
  it fails loudly if a blank ever appears with `tenure > 0`.
- **Mixed binary encodings.** `SeniorCitizen` is 0/1 while the other yes/no fields are
  `Yes`/`No`. A naive "encode the object columns" step would silently treat it as numeric.
- **"No internet/phone service" is a category, not a null.** It means the customer cannot have
  the add-on.
- **Class imbalance (26.5%).** Accuracy is misleading (73.5% by always predicting "stays"). The
  project reports ROC-AUC, F1, precision and recall on the churn class.
- **Static snapshot, no dates.** Change over time is simulated with batches. v1 holds the first
  5,000 rows and v2 adds the remaining 2,043 as "new signups" (26.3% vs 27.2% churn).
- **Sensitive attributes:** `gender` and `SeniorCitizen`. These are for later fairness checks,
  not features to optimise on blindly.
