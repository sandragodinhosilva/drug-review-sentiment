"""
Shared configuration constants and helpers for paths and tracking.

Centralizes the results directory so scripts and apps store logs, artifacts,
and MLflow runs in a single place.
"""
from __future__ import annotations

import os
from pathlib import Path

def _repo_root() -> Path:
    """Return the project root directory.

    Anchors paths to the repository root so notebooks or scripts executed from
    subdirectories do not create nested results directories.
    """
    # This file lives at <repo>/src/config.py, so parent of parent is repo root
    return Path(__file__).resolve().parents[1]


# Root results directory (can be overridden with env var)
# Use an absolute path anchored at the repo root by default
_DEFAULT_RESULTS_DIR = _repo_root() / "results"
RESULTS_DIR: str = os.getenv("RESULTS_DIR", str(_DEFAULT_RESULTS_DIR))

# Subdirectories and files
OUTPUTS_DIR: str = os.path.join(RESULTS_DIR, "outputs")
MLFLOW_DIR: str = os.path.join(RESULTS_DIR, "mlruns")
TRAIN_LOG: str = os.path.join(RESULTS_DIR, "train_model.log")
APP_LOG: str = os.path.join(RESULTS_DIR, "app.log")
TB_DIR: str = os.path.join(RESULTS_DIR, "tensorboard")


def ensure_dirs() -> None:
    """Create core results directories if they do not exist."""
    for p in (RESULTS_DIR, OUTPUTS_DIR, MLFLOW_DIR, TB_DIR):
        Path(p).mkdir(parents=True, exist_ok=True)


def mlflow_uri() -> str:
    """Return a file:// URI pointing to the MLflow tracking directory."""
    return f"file://{os.path.abspath(MLFLOW_DIR)}"
