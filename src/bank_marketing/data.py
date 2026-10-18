"""Download, validate, clean and split the UCI Bank Marketing dataset."""

from __future__ import annotations

import argparse
import io
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_marketing.config import Config, load_config
from bank_marketing.utils import (
    ensure_dir,
    save_json,
    setup_logging,
    sha256_bytes,
    sha256_file,
)

logger = logging.getLogger(__name__)

TARGET = "y"
POSITIVE_LABEL = "yes"
PDAYS_NOT_CONTACTED = 999 # sentinel: client was never contacted before
NUMERIC_COLUMNS: tuple[str, ...] = (
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
)
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)
RAW_COLUMNS: tuple[str, ...] = (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET)

def _levels(text: str) -> frozenset[str]:
    return frozenset(text.split())

# Documented levels (bank-additional-names.txt). "unknown" is real level here.
EXPECTED_CATEGORIES: dict[str, frozenset[str]] = {
    "job": _levels(
        "admin. blue-collar entrepreneur housemaid management retired "
        "self-employed services student technician unemployed unknown"
    ),
    "marital": _levels("divorced married single unknown"),
    "education": _levels(
        "basic.4y basic.6y basic.9y high.school illiterate "
        "professional.course university.degree unknown"
    ),
    "default": _levels("no yes unknown"),
    "housing": _levels("no yes unknown"),
    "loan": _levels("no yes unknown"),
    "contact": _levels("cellular telephone"),
    "month": _levels("jan feb mar apr may jun jul aug sep oct nov dec"),
    "day_of_week": _levels("mon tue wed thu fri"),
    "poutcome": _levels("failure nonexistent success"),
}

class DataValidationError(ValueError):
    """Raised when a dataframe does not match the expected schema."""
    
# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _find_csv_in_zip(payload: bytes, filename: str, depth: int = 0) -> bytes | None:
    """Return the CSV bytes from a (possibly nested) zip archive."""
    if depth > 2:
        return None
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            if member.startswith("__MACOSX/"):
                continue
            if Path(member).name == filename:
                return archive.read(member)
        for member in archive.namelist():
            if member.endswith(".zip") and not member.startswith("__MACOSX/"):
                found = _find_csv_in_zip(archive.read(member), filename, depth + 1)
                if found is not None:
                    return found 
    return None

def _extract_csv(payload: bytes, config: Config) -> bytes:
    filename = config.data.raw_filename
    if zipfile.is_zipfile(io.BytesIO(payload)):
        csv_bytes = _find_csv_in_zip(payload, filename)
        if csv_bytes is None:
            raise DataValidationError(f"{filename} not found inside the archive")
    else:
        csv_bytes = payload # a plain CSV was supplied
    expected = config.data.sha256
    actual = sha256_bytes(csv_bytes)
    if expected and actual != expected:
        raise DataValidationError(
            f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
        )
    return csv_bytes

def _fetch(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "bank-marketing-ml"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload: bytes = response.read()
    return payload

def download_raw(
    config: Config, *, source: Path | None = None, force: bool = False
) -> Path:
    """Place the verified raw CSV at ``config.data.raw_path``.
    
    ``source`` may be a manually downloaded zip or CSV file; otherwise the
    configured URLs are tried in order.
    """
    target = config.data.raw_path
    if target.exists() and not force:
        if config.data.sha256 and sha256_file(target) != config.data.sha256:
            raise DataValidationError(
                f"{target} exists but its checksum is wrong; rerun with --force"
            )
        logger.info("Raw data already present: %s", target)
        return target
    
    if source is not None:
        logger.info("Using local source file %s", source)
        csv_bytes = _extract_csv(source.read_bytes(), config)
    else:
        errors: list[str] = []
        csv_bytes = b""
        for url in config.data.urls:
            try:
                logger.info("Downloading %s", url)
                csv_bytes = _extract_csv(_fetch(url), config)
                break
            except (urllib.error.URLError, TimeoutError, DataValidationError) as exc:
                logger.warning("Source failed (%s): %s", url, exc)
                errors.append(f"{url}: {exc}")
        else:
            raise DataValidationError(
                "All download sources failed. Download the archive manually and "
                "run `bank-ml data download --source <file>`.\n" + "\n".join(errors)
            )
    
    ensure_dir(target.parent)
    target.write_bytes(csv_bytes)
    logger.info("Wrote %s (%d bytes)", target, len(csv_bytes))
    return target

# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Use snake_case names: 'emp.var.rate' -> 'emp_var_rate'."""
    renamed = df.rename(columns=lambda c: str(c).strip().replace(".", "_"))
    return renamed

def load_raw(path: Path) -> pd.DataFrame:
    """Read the semicolon separated raw CSV and normalize column names."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `bank-ml data download` first."
        )
    df = pd.read_csv(path, sep=";")
    return normalize_columns(df)

def check_schema(
    df: pd.DataFrame,
    *,
    required: tuple[str, ...] = RAW_COLUMNS,
    numeric: tuple[str, ...] = NUMERIC_COLUMNS,
) -> None:
    """Fail fast on missing columns or non-numeric numeric columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    bad_numeric = [
        c
        for c in numeric
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if bad_numeric:
        raise DataValidationError(f"Columns must be numeric: {bad_numeric}")
    if TARGET in df.columns:
        labels = set(df[TARGET].dropna().unique())
        if not labels <= {"yes", "no", 0, 1}:
            raise DataValidationError(f"Unexpected target labels: {sorted(labels)}")
        
def unexpected_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return category values that are not documented for each column."""
    result: dict[str, list[str]] = {}
    for column, allowed in EXPECTED_CATEGORIES.items():
        if column not in df.columns:
            continue
        values = set(df[column].dropna().astype(str).unique())
        extra = sorted(values - allowed)
        if extra:
            result[column] = extra
    return result

# --------------------------------------------------------------------------
# Cleaning and splitting
# --------------------------------------------------------------------------
def clean(
    df: pd.DataFrame, drop_columns: tuple[str, ...] = ("duration",)
) -> pd.DataFrame:
    """Apply the cleaning decisions motivated by the EDA.
    
    1. Drop exact duplicate records (identical in every raw column).
    2. Drop leakage columns such as ``duration``.
    3. Encode the target as 0/1.
    """
    check_schema(df)
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d exact duplicate rows", n_before - len(out))
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])
    out[TARGET] = (out[TARGET].astype(str) == POSITIVE_LABEL).astype("int64")
    return out

def split(
    df: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. The test part is locked until evaluation."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def prepare(config: Config) -> tuple[Path, Path]:
    """raw CSV -> cleaned, stratified train.csv / test.csv + manifest."""
    raw = load_raw(config.data.raw_path)
    problems = unexpected_categories(raw)
    if problems:
        logger.warning("Undocumented category values: %s", problems)
    cleaned = clean(raw, config.data.drop_columns)
    train_df, test_df = split(
        cleaned, test_size=config.data.test_size, seed=config.seed
    )
    
    out_dir = ensure_dir(config.data.processed_dir)
    train_df.to_csv(config.data.train_path, index=False)
    test_df.to_csv(config.data.test_path, index=False)
    manifest = {
        "raw_file": config.data.raw_path.as_posix(),
        "raw_sha256": sha256_file(config.data.raw_path),
        "seed": config.seed,
        "test_size": config.data.test_size,
        "dropped_columns": list(config.data.drop_columns),
        "rows": {"raw": len(raw), "clean": len(cleaned)},
        "train": {
            "rows": len(train_df),
            "positive_rate": float(train_df[TARGET].mean()),
        },
        "test": {"rows": len(test_df), "positive_rate": float(test_df[TARGET].mean())},
        "train_sha256": sha256_file(config.data.train_path),
        "test_sha256": sha256_file(config.data.test_path),
    }
    save_json(manifest, out_dir / "split_manifest.json")
    logger.info(
        "train=%d rows, test=%d rows, positive rate=%.4f",
        len(train_df),
        len(test_df),
        cleaned[TARGET].mean(),
    )
    return config.data.train_path, config.data.test_path

def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed split and return (X, y)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `bank-ml data prepare` first.")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise DataValidationError(f"{path} has no '{TARGET}' column")
    return df.drop(columns=[TARGET]), df[TARGET].astype("int64")

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bank-ml data", description=__doc__)
    parser.add_argument("action", choices=["download", "prepare", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", type=Path, help="local zip/CSV instead of URLs")
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    
    setup_logging(args.verbose)
    config = load_config(args.config)
    if args.action in {"download", "all"}:
        download_raw(config, source=args.source, force=args.force)
    if args.action in {"prepare", "all"}:
        prepare(config)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())