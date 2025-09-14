"""
Shared configuration constants and helpers for paths and tracking.

Centralizes the results directory so scripts and apps store logs, artifacts,
and MLflow runs in a single place.
"""
from __future__ import annotations

import os
from pathlib import Path

# Root results directory (override with env var if desired)
RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")

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
