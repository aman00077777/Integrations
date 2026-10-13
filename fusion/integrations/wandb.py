"""Weights & Biases integration for FUSION (optional dependency)."""

from __future__ import annotations

import logging
from typing import Any

try:
    import wandb
except ImportError as e:
    raise ImportError(
        "Weights & Biases is required for this integration. "
        "Install it with: pip install wandb"
    ) from e


logger = logging.getLogger("fusion.integrations.wandb")


class WandbLogger:
    """Log metrics, artifacts, and model information to Weights & Biases."""

    def __init__(
        self,
        project: str,
        config: Any,
        entity: str | None = None,
    ) -> None:
        """Initialize a Weights & Biases run."""
        self.run = wandb.init(
            project=project,
            config=config.to_dict(),
            entity=entity,
        )

    def log(self, metrics: dict, step: int) -> None:
        """Log metrics at the specified training step."""
        wandb.log(metrics, step=step)

    def log_artifact(
        self,
        path: str,
        name: str,
        type: str,
    ) -> None:
        """Log a file as a Weights & Biases artifact."""
        artifact = wandb.Artifact(
            name=name,
            type=type,
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def watch(self, model: Any) -> None:
        """Track gradients and weights of the model."""
        wandb.watch(model, log="all")


__all__ = ["WandbLogger"]