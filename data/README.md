# data/

| Path | In Git? | What |
| --- | --- | --- |
| `telco_churn.csv.dvc` | yes | DVC pointer: the md5 and size of the dataset version in use |
| `telco_churn.csv` | no (DVC) | the dataset. Get it with `uv run dvc pull` |
| `processed/` | no (DVC) | train/test split written by the `prepare` stage |
| `source/` | no (ignored) | the raw IBM download, only needed to rebuild a version |

Rebuild a dataset version from the IBM source (downloaded on first use):

```bash
uv run python src/main.py make-data --version 1   # first 5,000 customers
uv run python src/main.py make-data --version 2   # all 7,043 customers
uv run dvc add data/telco_churn.csv
```

The documentation for the columns is in [../docs/DATA_DICTIONARY.md](../docs/DATA_DICTIONARY.md).
