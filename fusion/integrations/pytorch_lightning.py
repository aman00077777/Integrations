"""
fusion/integrations/pytorch_lightning.py

Phase 15.1 — PyTorch Lightning integration for the FUSION framework.

Wraps a ``BaseFusionModel`` (or any ``nn.Module``) into a
``LightningModule`` so that users can train with PyTorch Lightning's
Trainer without manually wiring optimizers, schedulers, or logging.

Depends on:
    Phase 1  — Config
    Phase 5  — BaseFusionModel (forward contract)
    Phase 11 — Loss functions (BaseLoss and concrete subclasses)
    Phase 12 — build_optimizer
    Phase 13 — build_lr_scheduler
    Phase 14 — Metrics (compute_accuracy)
    Exceptions — ConfigError, FusionError

Owner: Aman Sharma
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import guard: pytorch_lightning / lightning.pytorch
#
# Because *this file* is named ``pytorch_lightning.py``, a bare
# ``import pytorch_lightning`` would resolve to ourselves.  We therefore
# use importlib to force-import the *real* installed library package and
# try both the legacy (``pytorch_lightning``) and the modern
# (``lightning.pytorch``) import paths.
# ---------------------------------------------------------------------------

_PL_MODULE = None  # will be set to the imported module

# Temporarily hide this file's own module from sys.modules so
# ``importlib.import_module`` doesn't short-circuit to ourselves.
_self_key = __name__  # "fusion.integrations.pytorch_lightning"
_saved_self = sys.modules.pop(_self_key, None)

try:
    try:
        _PL_MODULE = importlib.import_module("pytorch_lightning")
    except ImportError:
        _PL_MODULE = importlib.import_module("lightning.pytorch")
except ImportError:
    raise ImportError(
        "PyTorch Lightning is required for FusionLightningModule but is not "
        "installed.  Install it with:\n\n"
        "    pip install pytorch-lightning\n\n"
        "or, for the newer unified package:\n\n"
        "    pip install lightning\n"
    ) from None
finally:
    # Restore ourselves in sys.modules so later imports work normally.
    if _saved_self is not None:
        sys.modules[_self_key] = _saved_self

# Pull out the names we actually need.
LightningModule = _PL_MODULE.LightningModule

# ---------------------------------------------------------------------------
# FUSION framework imports
# ---------------------------------------------------------------------------
from fusion.exceptions import ConfigError
from fusion.training.optimizers import build_optimizer
from fusion.training.lr_scheduler import build_lr_scheduler
from fusion.training.metrics import compute_accuracy

logger = logging.getLogger("fusion.integrations.pytorch_lightning")


class FusionLightningModule(LightningModule):
    """LightningModule wrapper for any FUSION model.

    Bridges the gap between the FUSION training primitives (``build_optimizer``,
    ``build_lr_scheduler``, loss functions) and PyTorch Lightning's Trainer so
    that a FUSION model can be trained with a single ``Trainer.fit()`` call.

    Args:
        model: A ``nn.Module`` (typically a ``BaseFusionModel`` subclass)
            whose ``forward()`` is called during training / validation.
        config: A FUSION ``Config`` (or plain ``dict``) with at least an
            ``"optimizer"`` sub-dict.  Optional keys: ``"lr_scheduler"``,
            ``"loss"``.
        loss_fn: A pre-built loss function (``nn.Module``).  If *None*,
            ``config.get("loss")`` must be present and must contain a
            ``"type"`` key naming a class importable from ``fusion.losses``.

    Raises:
        ConfigError: If ``loss_fn`` is *None* and the config does not
            contain enough information to build one, or if the optimizer
            config is missing / invalid.

    Example::

        from fusion.utils.config import Config
        from fusion.integrations.pytorch_lightning import FusionLightningModule

        cfg = Config({
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "lr_scheduler": {"type": "cosine", "t_max": 10},
        })
        lit = FusionLightningModule(model, cfg, loss_fn=my_loss)
        trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
        trainer.fit(lit, train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.model = model
        # Normalise config: accept both a Config object and a plain dict.
        self._raw_config = config
        self._config_dict = (
            config.to_dict() if hasattr(config, "to_dict") else dict(config)
        )

        # ---- Loss function ---------------------------------------------------
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = self._build_loss_from_config()

        # ---- Optimizer / scheduler config (used in configure_optimizers) ------
        self._optimizer_config = self._resolve_sub_config("optimizer")
        if self._optimizer_config is None:
            raise ConfigError(
                "FusionLightningModule requires an 'optimizer' section in config.",
                details={"available_keys": list(self._config_dict.keys())},
            )

        self._scheduler_config = self._resolve_sub_config("lr_scheduler")

        # ---- save_hyperparameters (ignore the model — it's not serialisable) --
        try:
            self.save_hyperparameters(ignore=["model", "loss_fn"])
        except Exception:  # pragma: no cover – older PL versions
            pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped model's ``forward()``."""
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a single training step.

        Args:
            batch: Whatever the ``DataLoader`` yields.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            Scalar loss tensor (Lightning takes care of ``.backward()``).
        """
        loss = self._shared_step(batch, stage="train")
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a single validation step.

        Args:
            batch: Whatever the ``DataLoader`` yields.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        loss = self._shared_step(batch, stage="val")
        self.log(
            "val_loss", loss,
            on_step=False, on_epoch=True, prog_bar=True,
        )

        # --- optional accuracy metrics ------------------------------------
        self._log_val_metrics(batch)

        return loss

    # ------------------------------------------------------------------
    # Optimizers & schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        """Build the optimizer (and optionally scheduler) from the FUSION config.

        Returns:
            A single ``Optimizer``, or a dict of the form Lightning expects::

                {"optimizer": opt,
                 "lr_scheduler": {"scheduler": sched,
                                  "interval": "epoch",
                                  "monitor": "val_loss"}}
        """
        optimizer = build_optimizer(self.model, self._optimizer_config)

        if self._scheduler_config is None:
            return optimizer

        scheduler = build_lr_scheduler(optimizer, self._scheduler_config)

        sched_type = self._scheduler_config.get("type", "cosine")
        interval = self._scheduler_config.get("interval", "epoch")

        lr_scheduler_dict: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
        }

        # ReduceLROnPlateau requires a monitor metric.
        if sched_type == "reduce_on_plateau":
            lr_scheduler_dict["monitor"] = self._scheduler_config.get(
                "monitor", "val_loss"
            )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """Forward + loss computation shared by train / val steps.

        Handles three common batch formats:
            1. ``(inputs, targets)`` tuple/list
            2. ``dict`` with ``"inputs"`` and ``"targets"`` keys
            3. ``dict`` passed directly to both model and loss

        Args:
            batch: The raw batch from the DataLoader.
            stage: ``"train"`` or ``"val"`` (for future per-stage logic).

        Returns:
            Scalar loss tensor.
        """
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch.get("inputs", batch)
            targets = batch.get("targets", batch)
        else:
            inputs = batch
            targets = batch

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss

    def _log_val_metrics(self, batch: Any) -> None:
        """Compute and log validation metrics when possible.

        Silently skips if the batch/output format doesn't support
        accuracy computation (e.g. contrastive / retrieval tasks).
        """
        try:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("inputs", batch)
                targets = batch.get("targets", batch)
            else:
                return

            with torch.no_grad():
                outputs = self.model(inputs)

            # outputs may be a dict (BaseFusionModel) or a plain tensor.
            logits = outputs.get("logits", None) if isinstance(outputs, dict) else outputs
            labels = targets.get("labels", None) if isinstance(targets, dict) else targets

            if logits is not None and labels is not None and logits.dim() == 2:
                acc = compute_accuracy(logits, labels)
                self.log("val_accuracy", acc, on_epoch=True, prog_bar=True)
        except Exception:
            # Non-classification tasks — silently skip.
            pass

    def _resolve_sub_config(self, key: str) -> Any:
        """Extract a sub-config dict, supporting both Config objects and dicts.

        Returns:
            The sub-config as a dict-like object, or *None* if absent.
        """
        if hasattr(self._raw_config, "get"):
            val = self._raw_config.get(key, None)
        else:
            val = getattr(self._raw_config, key, None)

        if val is None:
            return None

        # Config objects need to be usable with .get(); convert if necessary.
        if hasattr(val, "to_dict"):
            return val.to_dict()
        if isinstance(val, dict):
            return val
        return val

    def _build_loss_from_config(self) -> nn.Module:
        """Instantiate a loss function from ``config.loss``.

        Looks up ``config.loss.type`` in ``fusion.losses`` and passes any
        remaining keys as constructor kwargs.

        Returns:
            An instantiated ``nn.Module`` loss.

        Raises:
            ConfigError: If the loss config is missing or the named class
                cannot be found / instantiated.
        """
        loss_cfg = self._resolve_sub_config("loss")
        if loss_cfg is None:
            raise ConfigError(
                "FusionLightningModule: no loss_fn provided and no 'loss' "
                "section found in config.",
                details={"available_keys": list(self._config_dict.keys())},
            )

        if not isinstance(loss_cfg, dict):
            loss_cfg = loss_cfg.to_dict() if hasattr(loss_cfg, "to_dict") else dict(loss_cfg)

        loss_type = loss_cfg.get("type", None)
        if loss_type is None:
            raise ConfigError(
                "FusionLightningModule: config.loss must have a 'type' key "
                "naming a class in fusion.losses (e.g. 'FocalLoss').",
                details={"loss_config": loss_cfg},
            )

        import fusion.losses as losses_module

        cls = getattr(losses_module, loss_type, None)
        if cls is None:
            raise ConfigError(
                f"FusionLightningModule: unknown loss type '{loss_type}'. "
                f"Available: {losses_module.__all__}",
                details={"requested": loss_type},
            )

        # Pass every key except 'type' as a constructor kwarg.
        kwargs = {k: v for k, v in loss_cfg.items() if k != "type"}
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise ConfigError(
                f"FusionLightningModule: failed to build '{loss_type}' with "
                f"kwargs {kwargs}: {exc}",
                details={"loss_type": loss_type, "kwargs": kwargs},
            ) from exc


__all__ = ["FusionLightningModule"]
