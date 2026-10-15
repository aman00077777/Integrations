"""MLflow integration for FUSION (optional dependency)."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional

try:
    import mlflow
except ImportError as e:
    raise ImportError(
        "MLflow is required for this integration. "
        "Install it with: pip install mlflow"
    ) from e


logger = logging.getLogger("fusion.integrations.mlflow")


class MLflowLogger:
    """Log parameters, metrics, and models to MLflow."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "fusion",
        run_name: Optional[str] = None,
    ) -> None:
        """Configure MLflow and start a run."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to the active run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at the specified training step."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Save a BaseFusionModel to a temp path and log it as an artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Directory-style save first, then file-style (e.g. torch.save).
                model.save(tmp_dir)
                mlflow.log_artifacts(tmp_dir, artifact_path=artifact_path)
            except (IsADirectoryError, PermissionError, OSError):
                file_path = os.path.join(tmp_dir, "model.pt")
                model.save(file_path)
                mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def finish(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()


__all__ = ["MLflowLogger"]
