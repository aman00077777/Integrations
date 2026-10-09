"""
tests/integrations/test_pytorch_lightning.py

Unit and integration tests for ``fusion.integrations.pytorch_lightning``.

Tests cover:
    - training_step returns a scalar loss
    - validation_step logs "val_loss"
    - configure_optimizers returns correct dict structure
    - Full 1-epoch Trainer.fit completes on CPU without errors
    - ImportError message when pytorch_lightning is missing

Run with:
    pytest tests/integrations/test_pytorch_lightning.py -v
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict
from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Minimal stub modules so that ``fusion.*`` imports work from this repo
# without needing the full Fusion- tree on sys.path.
# We only stub what pytorch_lightning.py actually imports.
# ---------------------------------------------------------------------------

def _ensure_fusion_stubs():
    """Register lightweight stub modules for the fusion dependencies
    that ``fusion.integrations.pytorch_lightning`` imports."""

    # --- fusion.exceptions ---------------------------------------------------
    if "fusion.exceptions" not in sys.modules:
        exc_mod = ModuleType("fusion.exceptions")

        class FusionError(Exception):
            def __init__(self, message, details=None):
                self.message = message
                self.details = details
                super().__init__(message)

        class ConfigError(FusionError):
            pass

        exc_mod.FusionError = FusionError
        exc_mod.ConfigError = ConfigError
        sys.modules["fusion.exceptions"] = exc_mod

    # --- fusion.training (package) -------------------------------------------
    if "fusion.training" not in sys.modules:
        sys.modules["fusion.training"] = ModuleType("fusion.training")

    # --- fusion.training.optimizers ------------------------------------------
    if "fusion.training.optimizers" not in sys.modules:
        opt_mod = ModuleType("fusion.training.optimizers")

        def build_optimizer(model, config):
            opt_type = config.get("type", "adamw") if hasattr(config, "get") else "adamw"
            lr = config.get("lr", 1e-3) if hasattr(config, "get") else 1e-3
            if opt_type == "sgd":
                return torch.optim.SGD(model.parameters(), lr=lr)
            return torch.optim.AdamW(model.parameters(), lr=lr)

        opt_mod.build_optimizer = build_optimizer
        sys.modules["fusion.training.optimizers"] = opt_mod

    # --- fusion.training.lr_scheduler ----------------------------------------
    if "fusion.training.lr_scheduler" not in sys.modules:
        sched_mod = ModuleType("fusion.training.lr_scheduler")

        def build_lr_scheduler(optimizer, config):
            sched_type = config.get("type", "step") if hasattr(config, "get") else "step"
            if sched_type == "reduce_on_plateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        sched_mod.build_lr_scheduler = build_lr_scheduler
        sys.modules["fusion.training.lr_scheduler"] = sched_mod

    # --- fusion.training.metrics ---------------------------------------------
    if "fusion.training.metrics" not in sys.modules:
        met_mod = ModuleType("fusion.training.metrics")

        def compute_accuracy(logits, labels):
            preds = torch.argmax(logits, dim=1)
            return (preds == labels).float().mean().item()

        met_mod.compute_accuracy = compute_accuracy
        sys.modules["fusion.training.metrics"] = met_mod

    # --- fusion.losses -------------------------------------------------------
    if "fusion.losses" not in sys.modules:
        loss_mod = ModuleType("fusion.losses")

        class BaseLoss(nn.Module):
            pass

        class FocalLoss(BaseLoss):
            def __init__(self, alpha=1.0, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, logits, targets):
                targets = targets.float()
                return nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="mean"
                )

        loss_mod.BaseLoss = BaseLoss
        loss_mod.FocalLoss = FocalLoss
        loss_mod.__all__ = ["BaseLoss", "FocalLoss"]
        sys.modules["fusion.losses"] = loss_mod


# Install stubs before importing the module under test.
_ensure_fusion_stubs()


# ---------------------------------------------------------------------------
# Now import the module under test.
# ---------------------------------------------------------------------------
import pytorch_lightning as pl
from fusion.integrations.pytorch_lightning import FusionLightningModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal model for testing: Linear(4 → 2)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: Any) -> torch.Tensor:
        if isinstance(x, dict):
            x = x.get("inputs", x.get("x"))
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self.linear(x)


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def tiny_config():
    return {
        "optimizer": {"type": "adamw", "lr": 1e-3},
        "lr_scheduler": {"type": "step", "step_size": 1},
    }


@pytest.fixture
def tiny_loss():
    return nn.MSELoss()


@pytest.fixture
def tiny_dataloader():
    """4 samples, input_dim=4, output_dim=2, batch_size=2."""
    x = torch.randn(4, 4)
    y = torch.randn(4, 2)
    return DataLoader(TensorDataset(x, y), batch_size=2)


@pytest.fixture
def lit_module(tiny_model, tiny_config, tiny_loss):
    return FusionLightningModule(
        model=tiny_model, config=tiny_config, loss_fn=tiny_loss,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainingStep:
    """training_step must return a scalar loss tensor."""

    def test_returns_scalar_loss(self, lit_module, tiny_dataloader):
        batch = next(iter(tiny_dataloader))
        loss = lit_module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0, "loss should be a scalar (0-dim tensor)"
        assert loss.requires_grad, "loss should be attached to the graph"

    def test_loss_is_finite(self, lit_module, tiny_dataloader):
        batch = next(iter(tiny_dataloader))
        loss = lit_module.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss), "loss should be finite"


class TestValidationStep:
    """validation_step must log 'val_loss'."""

    def test_logs_val_loss(self, lit_module, tiny_dataloader):
        batch = next(iter(tiny_dataloader))
        # Attach a fake trainer so self.log works (PL requirement).
        trainer = pl.Trainer(
            max_epochs=1, accelerator="cpu",
            logger=False, enable_checkpointing=False,
        )
        lit_module.trainer = trainer

        # Patch self.log to capture calls.
        logged: Dict[str, Any] = {}
        original_log = lit_module.log

        def capture_log(name, value, **kwargs):
            logged[name] = value
            try:
                original_log(name, value, **kwargs)
            except Exception:
                pass  # outside a training loop self.log may warn; that's fine

        lit_module.log = capture_log

        lit_module.validation_step(batch, batch_idx=0)
        assert "val_loss" in logged, f"Expected 'val_loss' in logged keys, got {list(logged.keys())}"
        assert isinstance(logged["val_loss"], torch.Tensor)


class TestConfigureOptimizers:
    """configure_optimizers must return the correct structure."""

    def test_with_scheduler(self, lit_module):
        result = lit_module.configure_optimizers()

        assert isinstance(result, dict), "Should return a dict when scheduler is configured"
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], torch.optim.Optimizer)

        sched_dict = result["lr_scheduler"]
        assert "scheduler" in sched_dict
        assert "interval" in sched_dict

    def test_without_scheduler(self, tiny_model, tiny_loss):
        config = {"optimizer": {"type": "adamw", "lr": 1e-3}}
        lit = FusionLightningModule(model=tiny_model, config=config, loss_fn=tiny_loss)
        result = lit.configure_optimizers()

        assert isinstance(result, torch.optim.Optimizer), \
            "Without scheduler, should return a bare Optimizer"

    def test_reduce_on_plateau_has_monitor(self, tiny_model, tiny_loss):
        config = {
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "lr_scheduler": {"type": "reduce_on_plateau"},
        }
        lit = FusionLightningModule(model=tiny_model, config=config, loss_fn=tiny_loss)
        result = lit.configure_optimizers()

        sched_dict = result["lr_scheduler"]
        assert "monitor" in sched_dict, "ReduceLROnPlateau needs a 'monitor' key"
        assert sched_dict["monitor"] == "val_loss"


class TestFullTrainerRun:
    """A 1-epoch Trainer.fit must finish without errors on CPU."""

    def test_one_epoch_cpu(self, tiny_model, tiny_config, tiny_loss, tiny_dataloader):
        lit = FusionLightningModule(
            model=tiny_model, config=tiny_config, loss_fn=tiny_loss,
        )
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        # Should not raise.
        trainer.fit(lit, train_dataloaders=tiny_dataloader, val_dataloaders=tiny_dataloader)

    def test_one_epoch_no_scheduler(self, tiny_model, tiny_loss, tiny_dataloader):
        config = {"optimizer": {"type": "adamw", "lr": 1e-2}}
        lit = FusionLightningModule(model=tiny_model, config=config, loss_fn=tiny_loss)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(lit, train_dataloaders=tiny_dataloader, val_dataloaders=tiny_dataloader)


class TestImportErrorMessage:
    """If pytorch_lightning is missing, a helpful ImportError must be raised."""

    def test_import_error_message(self):
        """Verify clear ImportError with install instructions appears when pytorch_lightning is missing."""
        import subprocess
        code = (
            "import sys\n"
            "sys.modules['pytorch_lightning'] = None\n"
            "sys.modules['lightning'] = None\n"
            "sys.modules['lightning.pytorch'] = None\n"
            "import fusion.integrations.pytorch_lightning\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Expected non-zero exit code due to missing pytorch_lightning"
        assert "pip install pytorch-lightning" in result.stderr, (
            f"Expected 'pip install pytorch-lightning' in stderr, got:\n{result.stderr}"
        )

    def test_import_error_message_in_process(self):
        """Verify in-process import error when patching sys.modules."""
        import importlib
        saved_module = sys.modules.get("fusion.integrations.pytorch_lightning")
        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "pytorch_lightning": None,
                    "lightning": None,
                    "lightning.pytorch": None,
                },
            ):
                sys.modules.pop("fusion.integrations.pytorch_lightning", None)
                with pytest.raises(ImportError, match="pip install pytorch-lightning"):
                    importlib.import_module("fusion.integrations.pytorch_lightning")
        finally:
            if saved_module is not None:
                sys.modules["fusion.integrations.pytorch_lightning"] = saved_module


class TestLossFromConfig:
    """Test auto-building a loss from config.loss when loss_fn is None."""

    def test_build_focal_loss_from_config(self, tiny_model):
        config = {
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "loss": {"type": "FocalLoss", "alpha": 0.5, "gamma": 1.0},
        }
        lit = FusionLightningModule(model=tiny_model, config=config, loss_fn=None)
        assert lit.loss_fn is not None
        assert type(lit.loss_fn).__name__ == "FocalLoss"

    def test_missing_loss_config_raises(self, tiny_model):
        config = {"optimizer": {"type": "adamw", "lr": 1e-3}}
        from fusion.exceptions import ConfigError
        with pytest.raises(ConfigError, match="no loss_fn provided"):
            FusionLightningModule(model=tiny_model, config=config, loss_fn=None)

    def test_unknown_loss_type_raises(self, tiny_model):
        config = {
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "loss": {"type": "NonExistentLoss"},
        }
        from fusion.exceptions import ConfigError
        with pytest.raises(ConfigError, match="unknown loss type"):
            FusionLightningModule(model=tiny_model, config=config, loss_fn=None)


class TestMissingOptimizerConfig:
    """Missing optimizer config must raise ConfigError."""

    def test_raises_config_error(self, tiny_model, tiny_loss):
        from fusion.exceptions import ConfigError
        with pytest.raises(ConfigError, match="optimizer"):
            FusionLightningModule(model=tiny_model, config={}, loss_fn=tiny_loss)
