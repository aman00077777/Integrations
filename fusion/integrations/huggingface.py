"""
fusion/integrations/huggingface.py

Phase 15.1 — Hugging Face Hub and Transformers integration for FUSION.

Provides capabilities to push and pull FUSION models to/from the
Hugging Face Hub and wrap FUSION configurations as Hugging Face
``PretrainedConfig`` objects.

Depends on:
    huggingface_hub
    transformers
    fusion.models.base (BaseFusionModel)
    fusion.utils.config (Config)
    fusion.exceptions (DeploymentError, FusionError)
    fusion.utils.logging (log_event, get_logger)
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import guard: huggingface_hub & transformers
# ---------------------------------------------------------------------------
try:
    import huggingface_hub
    import transformers
    from transformers import AutoConfig, PretrainedConfig
except ImportError:
    raise ImportError(
        "Hugging Face Hub and Transformers are required for this integration. "
        "Install them with: pip install huggingface_hub transformers"
    ) from None

# ---------------------------------------------------------------------------
# FUSION framework imports with resilient fallbacks
# ---------------------------------------------------------------------------
try:
    from fusion.exceptions import DeploymentError, FusionError
except ImportError:
    class FusionError(Exception):
        """Root exception for all Fusion-related errors."""

        def __init__(self, message: str, details: Optional[dict] = None) -> None:
            self.message = message
            self.details = details
            super().__init__(message)

        def __str__(self) -> str:
            if self.details:
                formatted = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
                return f"{self.message} [{formatted}]"
            return self.message

    class DeploymentError(FusionError):
        """Export, deployment, or Hugging Face Hub operation failure."""
        pass

try:
    from fusion.models.base import BaseFusionModel
except ImportError:
    class BaseFusionModel(nn.Module):
        """Base model stub for standalone environments."""

        def __init__(self, config: Optional[Any] = None) -> None:
            super().__init__()
            self.config = config

        def save(self, path: str) -> None:
            """Save model checkpoint."""
            pass

        @classmethod
        def load(cls, path: str) -> "BaseFusionModel":
            """Load model checkpoint."""
            return cls()

try:
    from fusion.utils.config import Config
except ImportError:
    class Config:
        """Recursive dictionary wrapper stub."""

        def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
            self._data = data or {}

        def to_dict(self) -> Dict[str, Any]:
            return dict(self._data)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "Config":
            return cls(data)

try:
    from fusion.utils.logging import get_logger, log_event
    logger = get_logger("integrations.huggingface")
except ImportError:
    logger = logging.getLogger("fusion.integrations.huggingface")

    def log_event(log: logging.Logger, level: int, event: str, **fields: Any) -> None:
        """Emit structured log event."""
        parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        log.log(level, f"event={event} {parts}".strip())


# ---------------------------------------------------------------------------
# Hugging Face Config Wrapper
# ---------------------------------------------------------------------------
class FusionHFConfig(PretrainedConfig):
    """HuggingFace PretrainedConfig wrapper for FUSION configurations.

    Wraps a native ``fusion.Config`` (or dictionary) so that FUSION models
    can seamlessly serialize configuration metadata to the Hugging Face Hub
    and support ``AutoConfig`` loading.

    Args:
        fusion_config: An optional FUSION ``Config`` object or dictionary.
        **kwargs: Additional parameters passed to ``PretrainedConfig``.
    """

    model_type = "fusion"

    def __init__(
        self,
        fusion_config: Optional[Union[Config, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if fusion_config is not None:
            if hasattr(fusion_config, "to_dict"):
                self.fusion_config = fusion_config.to_dict()
            elif isinstance(fusion_config, dict):
                self.fusion_config = dict(fusion_config)
            else:
                self.fusion_config = fusion_config
        else:
            self.fusion_config = kwargs.get("fusion_config", {})

        # Expose top-level fields of fusion_config as attributes if not already defined
        if isinstance(self.fusion_config, dict):
            for k, v in self.fusion_config.items():
                if not hasattr(self, k):
                    setattr(self, k, v)

    def to_fusion_config(self) -> Config:
        """Convert back to a native FUSION ``Config`` instance.

        Returns:
            A reconstructed ``Config`` object.
        """
        return Config.from_dict(self.fusion_config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain Python dictionary.

        Returns:
            Dictionary containing model configuration and HF metadata.
        """
        output = super().to_dict()
        output["model_type"] = self.model_type
        output["fusion_config"] = self.fusion_config
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs: Any) -> "FusionHFConfig":
        """Construct a ``FusionHFConfig`` from a Python dictionary.

        Args:
            config_dict: Dictionary containing configuration keys.
            **kwargs: Extra parameters passed to the constructor.

        Returns:
            A ``FusionHFConfig`` instance.
        """
        config_dict = dict(config_dict)
        fusion_config = config_dict.pop("fusion_config", None)
        return cls(fusion_config=fusion_config, **config_dict)


# Register with AutoConfig if available and not already registered
try:
    AutoConfig.register("fusion", FusionHFConfig)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Hub Operations
# ---------------------------------------------------------------------------
def push_to_hub(
    model: BaseFusionModel,
    repo_id: str,
    token: str,
    commit_message: Optional[str] = "Upload FUSION model",
    private: bool = False,
    **kwargs: Any,
) -> str:
    """Upload a BaseFusionModel to the Hugging Face Hub.

    Creates the remote repository if it does not exist, saves the model
    to a temporary directory via ``model.save()``, exports the HF configuration
    metadata, and uploads all files to the Hub.

    Args:
        model: The ``BaseFusionModel`` instance to save and upload.
        repo_id: Hugging Face repository identifier (e.g. ``'username/fusion-model'``).
        token: Hugging Face authentication token. Must not be empty.
        commit_message: Commit message for the upload commit.
        private: Whether to create the repository as private if creating anew.
        **kwargs: Additional keyword arguments forwarded to ``huggingface_hub.upload_folder``.

    Returns:
        The URL of the repository on the Hugging Face Hub.

    Raises:
        DeploymentError: If repo creation, model saving, or uploading fails.
    """
    if not token:
        raise DeploymentError(
            "A valid Hugging Face token is required to push to the Hub.",
            details={"repo_id": repo_id},
        )

    try:
        repo_url = huggingface_hub.create_repo(
            repo_id=repo_id,
            token=token,
            exist_ok=True,
            private=private,
            repo_type="model",
        )
    except Exception as exc:
        raise DeploymentError(
            f"Failed to create or access Hugging Face repository '{repo_id}': {exc}",
            details={"repo_id": repo_id},
        ) from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save model using BaseFusionModel's own save method
        try:
            try:
                # First try directory path directly
                model.save(tmp_dir)
            except (IsADirectoryError, PermissionError, OSError):
                # If model.save expects a file path (like torch.save), save to model.pt
                model.save(os.path.join(tmp_dir, "model.pt"))
        except Exception as exc:
            raise DeploymentError(
                f"Failed to save model to temporary directory before uploading: {exc}",
                details={"repo_id": repo_id},
            ) from exc

        # Export HuggingFace config metadata alongside weights if available
        try:
            if hasattr(model, "config") and model.config is not None:
                hf_config = FusionHFConfig(fusion_config=model.config)
                hf_config.save_pretrained(tmp_dir)
        except Exception:
            # Non-critical: metadata export failure does not invalidate model save
            pass

        try:
            huggingface_hub.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
                **kwargs,
            )
        except Exception as exc:
            raise DeploymentError(
                f"Failed to upload model folder to Hugging Face Hub '{repo_id}': {exc}",
                details={"repo_id": repo_id},
            ) from exc

    url = str(repo_url) if repo_url else f"https://huggingface.co/{repo_id}"
    log_event(logger, logging.INFO, "hf_hub_push_success", repo_id=repo_id, url=url)
    return url


def pull_from_hub(
    repo_id: str,
    revision: str = "main",
    token: Optional[str] = None,
    model_class: Optional[Type[BaseFusionModel]] = None,
    **kwargs: Any,
) -> BaseFusionModel:
    """Download and reconstruct a BaseFusionModel from the Hugging Face Hub.

    Downloads snapshot files for the specified repository and delegates
    reconstruction to ``BaseFusionModel.load()``.

    Args:
        repo_id: Hugging Face repository identifier (e.g. ``'username/fusion-model'``).
        revision: Specific Git commit, branch, or tag (defaults to ``'main'``).
        token: Optional authentication token (required for private repositories).
        model_class: The concrete ``BaseFusionModel`` subclass to instantiate.
            Defaults to ``BaseFusionModel``.
        **kwargs: Additional keyword arguments forwarded to ``huggingface_hub.snapshot_download``.

    Returns:
        The loaded ``BaseFusionModel`` instance.

    Raises:
        DeploymentError: If snapshot downloading or model loading fails.
    """
    try:
        downloaded_dir = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            **kwargs,
        )
    except Exception as exc:
        raise DeploymentError(
            f"Failed to download repository '{repo_id}' from Hugging Face Hub: {exc}",
            details={"repo_id": repo_id, "revision": revision},
        ) from exc

    target_cls = model_class if model_class is not None else BaseFusionModel

    try:
        # First attempt: load directly using downloaded directory
        try:
            return target_cls.load(downloaded_dir)
        except Exception:
            # If load expects a file path, look inside downloaded_dir for checkpoint files
            candidates = ["model.pt", "model.pth", "pytorch_model.bin", "checkpoint.pt"]
            for fname in candidates:
                fpath = os.path.join(downloaded_dir, fname)
                if os.path.isfile(fpath):
                    return target_cls.load(fpath)

            if os.path.isdir(downloaded_dir):
                for fname in os.listdir(downloaded_dir):
                    if fname.endswith((".pt", ".pth", ".bin")):
                        return target_cls.load(os.path.join(downloaded_dir, fname))
            raise
    except Exception as exc:
        raise DeploymentError(
            f"Failed to load BaseFusionModel from downloaded path '{downloaded_dir}': {exc}",
            details={"repo_id": repo_id, "downloaded_dir": downloaded_dir},
        ) from exc


__all__ = [
    "FusionHFConfig",
    "push_to_hub",
    "pull_from_hub",
]
