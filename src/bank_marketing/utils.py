"""Small helpers shared by all modules: logging, hashing, JSON, run metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
TRACKED_PACKAGES = ("scikit-learn", "pandas", "numpy", "joblib", "matplotlib")

def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once for CLI entry points."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def save_json(obj: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    return path

def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
    
def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def git_commit() -> str | None:
    """Commit that produced a run: $GIT_COMMIT (Docker builds) or `git rev-parse`."""
    from_env = os.environ.get("GIT_COMMIT", "").strip()
    if from_env:
        return from_env
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None

def environment_info() -> dict[str, Any]:
    """Versions that matter for reproducing a training run."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": git_commit(),
    }