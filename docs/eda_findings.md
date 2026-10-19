# EDA findings and modelling decisions

Numbers come from `bank-ml eda` (`reports/eda/quality_report.md`,
`reports/eda/quality_summary.json`, `reports/eda/target_rate_by_category.csv`)
and the notebook. Keep the decisions in sync with the code.

1. **Class imbalance.** Positive rate = 0.1127 (4640 / 41188).
   Decision: stratified split and CV; primary metric = average precision.
2. **Leakage.** `duration` differs strongly by target (median 163.5 vs
   449.0 s; ROC-AUC of `duration` alone = 0.818), but it is only known
   after the call. Decision: drop `duration` in `data.clean`.
3. **Sentinel value.** `pdays == 999` share = 0.9632.
   Decision: add `previously_contacted`, treat 999 as missing (`features.py`).
4. **"unknown" levels.** Rows with "unknown" per column:
   job 330, marital 80, education 1731, default 8597, housing 990, loan 990.
   Subscription rate for `default=unknown` = 0.052 vs
   `default=no` = 0.129 (`default=yes` has only 3 rows).
   Decision: keep `unknown` as its own category.
5. **Duplicates.** Exact duplicate rows = 12.
   Decision: drop them before the split.
6. **Correlated economic indicators.** Spearman correlation
   (emp_var_rate, euribor3m) = 0.940,
   (emp_var_rate, nr_employed) = 0.945.
   Decision: regularized logistic regression; do not interpret single
   coefficients; trees are unaffected.
7. **Skewed counts.** `campaign` skew = 4.76; subscription
   rate falls with more contacts. Decision: no transform for trees; scaling for LR.
8. **Time order.** Positive rate per block of 2000 consecutive rows ranges
   from 0.021 to 0.507. Decision: random split measures
   in-period performance only; temporal drift is documented in the model card.
