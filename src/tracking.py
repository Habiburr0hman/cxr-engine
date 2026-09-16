# src/tracking.py
import os

import mlflow

from src.config import AppConfig


def setup_mlflow(config: AppConfig) -> str:
    mode = config.project.tracking_mode
    has_credentials = bool(
        os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD")
    )
    if mode == "dagshub" and not has_credentials:
        raise OSError(
            "[ERROR] tracking_mode is set to 'dagshub', but MLFLOW_TRACKING_USERNAME "
            "or MLFLOW_TRACKING_PASSWORD is missing in your .env file or environment."
        )
    tracking_uri = config.active_tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)
    print(f"[*] MLflow Tracking ({mode.upper()}): {tracking_uri}")

    return tracking_uri
