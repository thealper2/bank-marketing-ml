# Dataset: UCI Bank Marketing (bank-additional-full)

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset 222 |
| DOI | https://doi.org/10.24432/C5K306 |
| Authors | S. Moro, P. Rita, P. Cortez (2014) |
| License | CC BY 4.0 (attribution required) |
| File | `bank-additional-full.csv` (semicolon separated) |
| SHA-256 | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` |
| Rows / inputs | 41,188 / 20 (+ target `y`) |
| Period | May 2008 - November 2010, rows ordered by date |
| Target | `y`: client subscribed a term deposit (`yes`/`no`) |

## Download strategy

`bank-ml data download` tries the URLs in `configs/default.yaml` in order and
accepts the first archive whose extracted CSV matches the SHA-256 above:

1. `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
    (outer zip containing `bank-additional.zip`)
2. `https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip`
    (legacy path)

If both fail, download the archive in a browser from the dataset page and run
`bank-ml data download --source path/to/archive.zip` (a plain CSV also works).
The checksum guarantees that every source yields byte-identical data.

## Columns

- Client: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- Last contact: `contact`, `month`, `day_of_week`, `duration`
- Campaign: `campaign`, `pdays` (999 = never contacted), `previous`, `poutcome`
- Economic context: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`,
  `euribor3m`, `nr.employed` (renamed to snake_case on load)

## Decisions

- `duration` is dropped: it is only known after the call and the dataset
  documentation states it should be discarded for realistic prediction.
- `unknown` is kept as a category level (it is informative, e.g. `default`).
- 12 exact duplicate rows are removed before splitting.
- Train/test split: stratified 80/20 with seed 42; test is used once.