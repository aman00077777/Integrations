"""
fusion/integrations/timm.py

Phase 15 — Integrations: timm (PyTorch Image Models) adapter.
Owner: Aman Hingawe

Provides:
    list_timm_models(filter_str) -> List[str]
    load_timm_encoder(model_name, pretrained=True) -> BaseEncoder

Depends on:
    timm (import-guarded per Phase 15 golden rule)
    Phase 2 — BaseEncoder, CustomVisionEncoder (resilient import with
    fallback stand-ins so this file stays importable standalone,
    matching the pattern used in huggingface.py).
    Verified against Phase 2 source: CustomVisionEncoder signature is
    (module, output_dim, config=None) — called positionally here.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for this integration. "
        "Install it with: pip install timm"
    ) from e

try:
    from fusion.exceptions import EncoderError
except ImportError:
    class FusionError(Exception):
        def __init__(self, message, details=None):
            super().__init__(message)
            self.message = message
            self.details = details

    class EncoderError(FusionError):
        pass

try:
    from fusion.encoders.base import BaseEncoder
except ImportError:
    import torch.nn as _nn

    class BaseEncoder(_nn.Module):
        """Minimal stand-in mirroring Phase 2 BaseEncoder contract."""

        def __init__(self, config=None):
            super().__init__()
            self.config = config or {}

        def encode(self, inputs):
            raise NotImplementedError

        def get_output_dim(self) -> int:
            raise NotImplementedError

try:
    from fusion.encoders.vision.custom import CustomVisionEncoder
except ImportError:
    class CustomVisionEncoder(BaseEncoder):
        """Stand-in mirroring Phase 2 CustomVisionEncoder(module, output_dim, config)."""

        def __init__(self, module, output_dim: int, config=None):
            super().__init__(config)
            self.module = module
            self._output_dim = int(output_dim)

        def encode(self, inputs):
            return self.module(inputs)

        def get_output_dim(self) -> int:
            return self._output_dim

try:
    from fusion.utils.logging import get_logger, log_event
    logger = get_logger("integrations.timm")
except ImportError:
    logger = logging.getLogger("fusion.integrations.timm")

    def log_event(log, level, event, **fields):
        parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        log.log(level, f"event={event} {parts}".strip())


def list_timm_models(filter_str: str = "") -> List[str]:
    """List available timm model names, optionally filtered by wildcard."""
    try:
        return sorted(timm.list_models(filter_str))
    except Exception as exc:
        raise EncoderError(
            f"Failed to list timm models with filter {filter_str!r}: {exc}",
            details={"filter": filter_str},
        ) from exc


def load_timm_encoder(
    model_name: str,
    pretrained: bool = True,
    config: Optional[Any] = None,
) -> BaseEncoder:
    """Load a timm model as a feature extractor wrapped in CustomVisionEncoder.

    Creates the model with num_classes=0 (classification head removed),
    reads the feature dimension from .num_features (fallback:
    .feature_info.channels()[-1]), and wraps it in Phase 2's
    CustomVisionEncoder (positional call: module, output_dim, config).
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    except Exception as exc:
        raise EncoderError(
            f"Failed to create timm model {model_name!r}: {exc}",
            details={"model_name": model_name, "pretrained": pretrained},
        ) from exc

    output_dim = getattr(model, "num_features", None)
    if output_dim is None:
        try:
            output_dim = model.feature_info.channels()[-1]
        except Exception as exc:
            raise EncoderError(
                f"Could not determine output dim for {model_name!r}",
                details={"model_name": model_name},
            ) from exc

    log_event(
        logger, logging.INFO, "timm_encoder_loaded",
        model=model_name, pretrained=pretrained, output_dim=int(output_dim),
    )
    return CustomVisionEncoder(model, int(output_dim), config)


__all__ = ["list_timm_models", "load_timm_encoder"]