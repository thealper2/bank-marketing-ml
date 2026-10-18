"""Baseline training (v1): numeric features only, Dummy vs LogisticRegression.

Phase 4 replaces this file with the full pipeline (all features, model
comparison, hyperparameter search, threshold selection, model bundle).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bank_marketing.config import Config, load_config
from bank_marketing.data import load_split
from bank_marketing.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

BASELINE_FEATURES = (
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
)

def baseline_models(seed: int) -> dict[str, object]:
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
    }
    
def run_baseline(config: Config) -> pd.DataFrame:
    X, y = load_split(config.data.train_path)
    X = X[list(BASELINE_FEATURES)]
    cv = StratifiedKFold(
        n_splits=config.cv.n_splits, shuffle=True, random_state=config.seed
    )
    rows = []
    
    for name, estimator in baseline_models(config.seed).items():
        scores = cross_validate(
            estimator, X, y, cv=cv, scoring=["average_precision", "roc_auc"]
        )
        rows.append(
            {
                "model": name,
                "average_precision_mean": scores["test_average_precision"].mean(),
                "average_precision_std": scores["test_average_precision"].std(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )
    table = pd.DataFrame(rows)
    out = ensure_dir(config.paths.metrics_dir) / "baseline_cv.csv"
    table.to_csv(out, index=False)
    logger.info("Baseline results:\n%s", table.to_string(index=False))
    return table

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bank-ml train", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    
    setup_logging(args.verbose)
    run_baseline(load_config(args.config))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())