"""Exploratory data analysis and data quality report for the raw dataset.

Runs on the *raw* file (before cleaning) so that duplicates, sentinels and
leakage columns are visible. Outputs go to reports/eda and reports/figures.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bank_marketing.config import Config, load_config
from bank_marketing.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PDAYS_NOT_CONTACTED,
    TARGET,
    check_schema,
    load_raw,
    unexpected_categories,
)
from bank_marketing.utils import ensure_dir, save_json, setup_logging

logger = logging.getLogger(__name__)

def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Machine-readable data quality facts."""
    target = (df[TARGET] == "yes").astype(int)
    unknown_counts = {
        c: int((df[c] == "unknown").sum())
        for c in CATEGORICAL_COLUMNS
        if (df[c] == "unknown").any()
    }
    numeric = df[list(NUMERIC_COLUMNS)]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outside = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
    features_only = df.drop(columns=[TARGET, "duration"], errors="ignore")
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_nan": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "unknown_level_counts": unknown_counts,
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows_without_duration": int(
        features_only.duplicated().sum()
        ),
        "target_counts": df[TARGET].value_counts().to_dict(),
        "positive_rate": float(target.mean()),
        "pdays_sentinel_share": float((df["pdays"] == PDAYS_NOT_CONTACTED).mean()),
        "iqr_outlier_share": {c: float(v) for c, v in outside.mean().round(4).items()},
        "numeric_skew": {c: float(v) for c, v in numeric.skew().round(3).items()},
        "undocumented_categories": unexpected_categories(df),
    }

def target_rate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency and subscription rate for every categorical level."""
    target = (df[TARGET] == "yes").astype(int)
    frames = []
    for column in CATEGORICAL_COLUMNS:
        grouped = (
            pd.DataFrame({"level": df[column], "y": target})
            .groupby("level")
            .agg(count=("y", "size"), positive_rate=("y", "mean"))
            .reset_index()
        )
        grouped.insert(0, "feature", column)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)

def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations (robust to skew) incl. the binary target."""
    numeric = df[list(NUMERIC_COLUMNS)].copy()
    numeric[TARGET] = (df[TARGET] == "yes").astype(int)
    return numeric.corr(method="spearman").round(3)

def _save(fig: Any, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

def save_figures(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    ensure_dir(out_dir)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(4, 3))
    df[TARGET].value_counts().plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Target distribution (y)")
    ax.set_ylabel("rows")
    paths.append(_save(fig, out_dir / "eda_target_distribution.png"))

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, column in zip(axes.ravel(), NUMERIC_COLUMNS, strict=True):
        ax.hist(df[column], bins=40, color="#4C72B0")
        ax.set_title(column)
    paths.append(_save(fig, out_dir / "eda_numeric_histograms.png"))

    rates = target_rate_by_category(df)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, column in zip(axes.ravel(), CATEGORICAL_COLUMNS, strict=True):
        part = rates[rates["feature"] == column].sort_values("positive_rate")
        ax.barh(part["level"].astype(str), part["positive_rate"], color="#55A868")
        ax.set_title(f"P(y=yes) by {column}")
    paths.append(_save(fig, out_dir / "eda_target_rate_by_category.png"))

    corr = numeric_correlations(df)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.index)
    fig.colorbar(image, ax=ax)
    ax.set_title("Spearman correlation")
    paths.append(_save(fig, out_dir / "eda_correlation.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    groups = [df.loc[df[TARGET] == label, "duration"] for label in ("no", "yes")]
    ax.boxplot(groups, tick_labels=["no", "yes"], showfliers=False)
    ax.set_title("duration by target (leakage check)")
    paths.append(_save(fig, out_dir / "eda_duration_leakage.png"))

    fig, ax = plt.subplots(figsize=(5, 4))
    capped = df["campaign"].clip(upper=15)
    rate = (df[TARGET] == "yes").groupby(capped).mean()
    ax.plot(rate.index, rate.to_numpy(), marker="o")
    ax.set_xlabel("campaign (capped at 15)")
    ax.set_ylabel("P(y=yes)")
    ax.set_title("Subscription rate vs. number of contacts")
    paths.append(_save(fig, out_dir / "eda_campaign_rate.png"))
    return paths

def render_markdown(summary: dict[str, Any], corr: pd.DataFrame) -> str:
    features = [c for c in corr.columns if c != TARGET]
    values = corr.loc[features, features].to_numpy()
    pairs = [
        (a, b, float(values[i, j]))
        for i, a in enumerate(features)
        for j, b in enumerate(features)
        if j > i
    ]
    top_pairs = sorted(pairs, key=lambda p: abs(p[2]), reverse=True)[:5]
    lines = [
        "# Data quality report",
        "",
        f"- Rows: {summary['n_rows']}, columns: {summary['n_columns']}",
        f"- Positive rate (y=yes): {summary['positive_rate']:.4f}",
        f"- Exact duplicate rows: {summary['exact_duplicate_rows']}",
        f"- NaN values: {summary['missing_nan'] or 'none'}",
        f"- 'unknown' levels: {summary['unknown_level_counts']}",
        f"- pdays == 999 share: {summary['pdays_sentinel_share']:.4f}",
        f"- Undocumented categories: {summary['undocumented_categories'] or 'none'}",
        "",
        "## Strongest numeric correlations (|Spearman|)",
        "",
    ]
    lines += [f"- {a} / {b}: {v:.3f}" for a, b, v in top_pairs]
    lines += ["", "## Correlation with target", ""]
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    lines += [f"- {name}: {value:.3f}" for name, value in target_corr.items()]
    return "\n".join(lines) + "\n"

def run_eda(config: Config) -> dict[str, Any]:
    df = load_raw(config.data.raw_path)
    check_schema(df)
    out_dir = ensure_dir(config.paths.reports_dir / "eda")

    summary = quality_summary(df)
    corr = numeric_correlations(df)
    save_json(summary, out_dir / "quality_summary.json")
    target_rate_by_category(df).to_csv(
        out_dir / "target_rate_by_category.csv", index=False
    )
    df[list(NUMERIC_COLUMNS)].describe().T.to_csv(out_dir / "numeric_summary.csv")
    corr.to_csv(out_dir / "numeric_correlations.csv")
    (out_dir / "quality_report.md").write_text(
        render_markdown(summary, corr), encoding="utf-8"
    )
    figures = save_figures(df, config.paths.figures_dir)
    logger.info("EDA written to %s (%d figures)", out_dir, len(figures))
    return summary

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bank-ml eda", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    
    setup_logging(args.verbose)
    run_eda(load_config(args.config))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
