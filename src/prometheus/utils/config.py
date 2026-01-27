from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # MLflow tracking
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_model_name: str

    # Storage in duckdb for inference logs and feedback labels
    duckdb_path: str

    # Monitoring parameter defaults
    drift_window_size: int
    drift_psi_threshold: float


def get_settings() -> Settings:
    
    # Load settings from environment variablesocal runs and Docker runs consistent
    
    load_dotenv()  # For testing with example .env

    return Settings(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", "prometheus_payment_risk"),
        mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "prometheus_payment_risk"),
        
        duckdb_path=os.getenv("DUCKDB_PATH", "./data/prometheus.duckdb"),
        drift_window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "5000")),
        drift_psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2")),
    )