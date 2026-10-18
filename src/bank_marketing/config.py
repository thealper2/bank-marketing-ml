"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""
    
@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    raw_filename: str = "bank-additional-full.csv"
    sha256: str | None = None
    urls: tuple[str, ...] = ()
    test_size: float = 0.2
    drop_columns: tuple[str, ...] = ("duration",)
    
    @property
    def raw_path(self) -> Path:
        return self.raw_dir / self.raw_filename
    
    @property
    def train_path(self) -> Path:
        return self.processed_dir / "train.csv"
    
    @property
    def test_path(self) -> Path:
        return self.processed_dir / "test.csv"
    
@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    
@dataclass(frozen=True)
class ModelsConfig:
    compare: tuple[str, ...] = (
        "dummy",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    )

@dataclass(frozen=True)
class SearchConfig:
    model: str = "hist_gradient_boosting"
    n_iter: int = 25
    scoring: str = "average_precision"
    n_jobs: int = -1
    
@dataclass(frozen=True)
class ThresholdConfig:
    strategy: str = "max_f1"
    min_precision: float = 0.5
    
@dataclass(frozen=True)
class PathsConfig:
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    
    @property
    def model_path(self) -> Path:
        return self.models_dir / "model.joblib"
    
    @property
    def metrics_dir(self) -> Path:
        return self.reports_dir / "metrics"
    
    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"
    
@dataclass(frozen=True)
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML friendly representation (paths as strings)."""
        
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list | tuple):
                return [convert(v) for v in value]
            return value
        
        result: dict[str, Any] = convert(asdict(self))
        return result
    
def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(value)}")
    return value

def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a validated Config from a plain dictionary."""
    known = {"seed", "data", "cv", "models", "search", "threshold", "paths"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown top-level config keys: {sorted(unknown)}")
    
    try:
        data_raw = _section(raw, "data")
        data = DataConfig(
            raw_dir=Path(data_raw.get("raw_dir", "data/raw")),
            processed_dir=Path(data_raw.get("processed_dir", "data/processed")),
            raw_filename=data_raw.get("raw_filename", "bank-additional-full.csv"),
            sha256=data_raw.get("sha256"),
            urls=tuple(data_raw.get("urls", ())),
            test_size=float(data_raw.get("test_size", 0.2)),
            drop_columns=tuple(data_raw.get("drop_columns", ("duration",))),
        )
        paths_raw = _section(raw, "paths")
        config = Config(
            seed=int(raw.get("seed", 42)),
            data=data,
            cv=CVConfig(**_section(raw, "cv")),
            models=ModelsConfig(
                compare=tuple(
                    _section(raw, "models").get("compare", ModelsConfig().compare)
                )
            ),
            search=SearchConfig(**_section(raw, "search")),
            threshold=ThresholdConfig(**_section(raw, "threshold")),
            paths=PathsConfig(
                models_dir=Path(paths_raw.get("models_dir", "models")),
                reports_dir=Path(paths_raw.get("reports_dir", "reports")),
            ),
        )
    except TypeError as exc: # unexpected keys inside a section
        raise ConfigError(f"Invalid configuration: {exc}") from exc
    
    _validate(config)
    return config

def _validate(config: Config) -> None:
    if not 0.0 < config.data.test_size < 0.5:
        raise ConfigError("data.test_size must be in (0, 0.5)")
    if config.cv.n_splits < 2:
        raise ConfigError("cv.n_splits must be >= 2")
    if config.search.n_iter < 1:
        raise ConfigError("search.n_iter must be >= 1")
    if config.threshold.strategy not in {"max_f1", "min_precision"}:
        raise ConfigError("threshold.strategy must be 'max_f1' or 'min_precision'")
    if not 0.0 < config.threshold.min_precision <= 1.0:
        raise ConfigError("threshold.min_precision must be in (0, 1]")
    
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return config_from_dict(raw)
