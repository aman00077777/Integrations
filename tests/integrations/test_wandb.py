"""Tests for the optional Weights & Biases integration."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


def _install_fake_wandb():
    """Install a lightweight fake wandb module for isolated tests."""
    fake_wandb = ModuleType("wandb")

    fake_wandb.init = mock.Mock()
    fake_wandb.log = mock.Mock()
    fake_wandb.log_artifact = mock.Mock()
    fake_wandb.watch = mock.Mock()

    artifact = mock.Mock()
    fake_wandb.Artifact = mock.Mock(return_value=artifact)

    sys.modules["wandb"] = fake_wandb
    return fake_wandb, artifact


@pytest.fixture
def fake_wandb():
    """Provide a fake wandb module and reload the integration."""
    fake_module, artifact = _install_fake_wandb()

    sys.modules.pop("fusion.integrations.wandb", None)
    module = importlib.import_module("fusion.integrations.wandb")

    yield fake_module, artifact, module

    sys.modules.pop("fusion.integrations.wandb", None)


class TestWandbLogger:
    """Tests for WandbLogger behavior."""

    def test_init_calls_wandb_init(self, fake_wandb):
        """Verify initialization creates a W&B run with the correct config."""
        wandb, _, module = fake_wandb

        config = SimpleNamespace(
            to_dict=mock.Mock(return_value={"lr": 0.001, "batch_size": 32})
        )

        module.WandbLogger(
            project="fusion-project",
            config=config,
            entity="test-user",
        )

        wandb.init.assert_called_once_with(
            project="fusion-project",
            config={"lr": 0.001, "batch_size": 32},
            entity="test-user",
        )

        config.to_dict.assert_called_once_with()

    def test_log_calls_wandb_log(self, fake_wandb):
        """Verify metrics are logged with the requested step."""
        wandb, _, module = fake_wandb

        config = SimpleNamespace(
            to_dict=lambda: {}
        )

        logger = module.WandbLogger(
            project="fusion-project",
            config=config,
        )

        metrics = {
            "train/loss": 0.42,
            "train/accuracy": 0.91,
        }

        logger.log(metrics, step=10)

        wandb.log.assert_called_once_with(
            metrics,
            step=10,
        )

    def test_log_artifact(self, fake_wandb, tmp_path):
        """Verify a file is added to and logged as a W&B artifact."""
        wandb, artifact, module = fake_wandb

        config = SimpleNamespace(
            to_dict=lambda: {}
        )

        logger = module.WandbLogger(
            project="fusion-project",
            config=config,
        )

        file_path = tmp_path / "model.pt"
        file_path.write_text("test model")

        logger.log_artifact(
            path=str(file_path),
            name="test-model",
            type="model",
        )

        wandb.Artifact.assert_called_once_with(
            name="test-model",
            type="model",
        )

        artifact.add_file.assert_called_once_with(
            str(file_path)
        )

        wandb.log_artifact.assert_called_once_with(artifact)

    def test_watch_calls_wandb_watch(self, fake_wandb):
        """Verify model watching is enabled with log='all'."""
        wandb, _, module = fake_wandb

        config = SimpleNamespace(
            to_dict=lambda: {}
        )

        logger = module.WandbLogger(
            project="fusion-project",
            config=config,
        )

        model = object()

        logger.watch(model)

        wandb.watch.assert_called_once_with(
            model,
            log="all",
        )


class TestImportGuard:
    """Tests for the optional dependency import guard."""

    def test_import_error_message(self):
        """Verify a clear installation message is raised when W&B is missing."""
        saved_module = sys.modules.get("fusion.integrations.wandb")

        try:
            sys.modules.pop("fusion.integrations.wandb", None)

            with mock.patch.dict(sys.modules, {"wandb": None}):
                with pytest.raises(
                    ImportError,
                    match="pip install wandb",
                ):
                    importlib.import_module(
                        "fusion.integrations.wandb"
                    )
        finally:
            sys.modules.pop("fusion.integrations.wandb", None)

            if saved_module is not None:
                sys.modules["fusion.integrations.wandb"] = saved_module