"""
tests/integrations/test_huggingface.py

Unit and integration tests for ``fusion.integrations.huggingface``
and the shared ``require`` helper in ``fusion.integrations``.

Tests:
    - push_to_hub calls create_repo with exist_ok=True and uploads folder
    - pull_from_hub returns a BaseFusionModel instance
    - FusionHFConfig round-trips to_dict / from_dict correctly
    - ImportError message appears when huggingface_hub is missing
    - Optional live test skipped when HF_TOKEN is not set
    - require helper raises clear instructions on missing dependencies
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import ModuleType
from typing import Any, Dict, Optional
from unittest import mock

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure lightweight stubs for core FUSION dependencies so tests
# run isolated without requiring the entire multi-phase codebase.
# ---------------------------------------------------------------------------
def _ensure_fusion_stubs():
    if "fusion.exceptions" not in sys.modules:
        exc_mod = ModuleType("fusion.exceptions")

        class FusionError(Exception):
            def __init__(self, message: str, details: Optional[dict] = None):
                self.message = message
                self.details = details
                super().__init__(message)

        class DeploymentError(FusionError):
            pass

        class ConfigError(FusionError):
            pass

        class CheckpointError(FusionError):
            pass

        exc_mod.FusionError = FusionError
        exc_mod.DeploymentError = DeploymentError
        exc_mod.ConfigError = ConfigError
        exc_mod.CheckpointError = CheckpointError
        sys.modules["fusion.exceptions"] = exc_mod

    if "fusion.utils" not in sys.modules:
        sys.modules["fusion.utils"] = ModuleType("fusion.utils")

    if "fusion.utils.config" not in sys.modules:
        cfg_mod = ModuleType("fusion.utils.config")

        class Config:
            def __init__(self, data: Optional[Dict[str, Any]] = None):
                self._data = dict(data) if data else {}
                for k, v in self._data.items():
                    setattr(self, k, v)

            def to_dict(self) -> Dict[str, Any]:
                return dict(self._data)

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> "Config":
                return cls(data)

        cfg_mod.Config = Config
        sys.modules["fusion.utils.config"] = cfg_mod

    if "fusion.utils.logging" not in sys.modules:
        log_mod = ModuleType("fusion.utils.logging")
        import logging

        def get_logger(name: str):
            return logging.getLogger(f"fusion.{name}")

        def log_event(logger, level, event, **fields):
            parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
            logger.log(level, f"event={event} {parts}".strip())

        log_mod.get_logger = get_logger
        log_mod.log_event = log_event
        sys.modules["fusion.utils.logging"] = log_mod

    if "fusion.models" not in sys.modules:
        sys.modules["fusion.models"] = ModuleType("fusion.models")

    if "fusion.models.base" not in sys.modules:
        base_mod = ModuleType("fusion.models.base")

        class BaseFusionModel(nn.Module):
            def __init__(self, config=None):
                super().__init__()
                self.config = config

            def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
                return {"logits": torch.zeros(1, 1)}

            def save(self, path: str) -> None:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                torch.save({"config": getattr(self.config, "to_dict", lambda: {})()}, path)

            @classmethod
            def load(cls, path: str) -> "BaseFusionModel":
                return cls()

        base_mod.BaseFusionModel = BaseFusionModel
        sys.modules["fusion.models.base"] = base_mod


_ensure_fusion_stubs()

# Imports under test
from fusion.exceptions import DeploymentError
from fusion.integrations import require
from fusion.integrations.huggingface import (
    FusionHFConfig,
    pull_from_hub,
    push_to_hub,
)
from fusion.models.base import BaseFusionModel
from fusion.utils.config import Config


# ---------------------------------------------------------------------------
# Dummy model implementation for tests
# ---------------------------------------------------------------------------
class DummyFusionModel(BaseFusionModel):
    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__(config=config)
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = inputs["x"]
        return {"logits": self.fc(x)}

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            file_path = os.path.join(path, "model.pt")
        else:
            file_path = path
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
                "class_name": type(self).__name__,
            },
            file_path,
        )

    @classmethod
    def load(cls, path: str) -> "DummyFusionModel":
        if os.path.isdir(path):
            file_path = os.path.join(path, "model.pt")
        else:
            file_path = path
        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ckpt.get("config", {}))
        model = cls(config=cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------------------
# Tests: Shared Import Guard (require)
# ---------------------------------------------------------------------------
class TestRequireHelper:
    def test_require_existing_module(self):
        math_mod = require("math")
        assert math_mod is not None
        assert hasattr(math_mod, "sqrt")

    def test_require_missing_module_raises(self):
        with pytest.raises(ImportError) as exc_info:
            require("non_existent_pkg_12345", "my-cool-package")
        msg = str(exc_info.value)
        assert "is required for this integration" in msg
        assert "pip install my-cool-package" in msg

    def test_require_missing_module_default_pip_name(self):
        with pytest.raises(ImportError) as exc_info:
            require("non_existent_pkg_67890")
        msg = str(exc_info.value)
        assert "pip install non_existent_pkg_67890" in msg


# ---------------------------------------------------------------------------
# Tests: FusionHFConfig
# ---------------------------------------------------------------------------
class TestFusionHFConfig:
    def test_model_type(self):
        cfg = FusionHFConfig()
        assert cfg.model_type == "fusion"

    def test_round_trip_dict(self):
        raw_config = {
            "hidden_dim": 256,
            "fusion_type": "cross_attention",
            "modalities": ["vision", "language"],
        }
        cfg = FusionHFConfig(fusion_config=raw_config)
        assert cfg.fusion_config == raw_config
        assert cfg.hidden_dim == 256
        assert cfg.fusion_type == "cross_attention"

        # Serialize to dict
        d = cfg.to_dict()
        assert d["model_type"] == "fusion"
        assert d["fusion_config"] == raw_config

        # Reconstruct from dict
        cfg2 = FusionHFConfig.from_dict(d)
        assert cfg2.model_type == "fusion"
        assert cfg2.fusion_config == raw_config

    def test_to_fusion_config(self):
        raw_config = {"lr": 0.001, "batch_size": 32}
        cfg = FusionHFConfig(fusion_config=raw_config)
        fusion_cfg = cfg.to_fusion_config()
        assert hasattr(fusion_cfg, "to_dict")
        assert fusion_cfg.to_dict() == raw_config


# ---------------------------------------------------------------------------
# Tests: push_to_hub
# ---------------------------------------------------------------------------
class TestPushToHub:
    @mock.patch("huggingface_hub.upload_folder")
    @mock.patch("huggingface_hub.create_repo")
    def test_push_calls_create_repo_and_upload(self, mock_create_repo, mock_upload_folder):
        mock_create_repo.return_value = "https://huggingface.co/username/test-model"
        folder_checked = []

        def fake_upload(*args, **kwargs):
            folder_path = kwargs.get("folder_path")
            folder_checked.append(os.path.isdir(folder_path))

        mock_upload_folder.side_effect = fake_upload

        cfg = Config({"dim": 4, "type": "dummy"})
        model = DummyFusionModel(config=cfg)
        repo_id = "username/test-model"
        token = "fake_hf_token_123"

        url = push_to_hub(model=model, repo_id=repo_id, token=token)

        # Verify create_repo was called with exist_ok=True
        mock_create_repo.assert_called_once_with(
            repo_id=repo_id,
            token=token,
            exist_ok=True,
            private=False,
            repo_type="model",
        )

        # Verify upload_folder was called with token and repo_id
        mock_upload_folder.assert_called_once()
        call_kwargs = mock_upload_folder.call_args[1]
        assert call_kwargs["repo_id"] == repo_id
        assert call_kwargs["token"] == token
        assert folder_checked == [True], "Folder should exist during upload"

        # Verify returned URL
        assert url == "https://huggingface.co/username/test-model"

    def test_push_empty_token_raises_deployment_error(self):
        model = DummyFusionModel(config=Config({}))
        with pytest.raises(DeploymentError, match="valid Hugging Face token"):
            push_to_hub(model=model, repo_id="user/repo", token="")

    @mock.patch("huggingface_hub.create_repo", side_effect=RuntimeError("Network down"))
    def test_push_create_repo_failure_raises_deployment_error(self, mock_create):
        model = DummyFusionModel(config=Config({}))
        with pytest.raises(DeploymentError, match="Failed to create or access"):
            push_to_hub(model=model, repo_id="user/repo", token="fake_token")


# ---------------------------------------------------------------------------
# Tests: pull_from_hub
# ---------------------------------------------------------------------------
class TestPullFromHub:
    @mock.patch("huggingface_hub.snapshot_download")
    def test_pull_returns_base_fusion_model(self, mock_download):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save a dummy model to that directory
            cfg = Config({"dim": 4})
            saved_model = DummyFusionModel(config=cfg)
            saved_model.save(os.path.join(tmp_dir, "model.pt"))

            mock_download.return_value = tmp_dir

            loaded_model = pull_from_hub(
                repo_id="username/test-model",
                revision="main",
                model_class=DummyFusionModel,
            )

            mock_download.assert_called_once_with(
                repo_id="username/test-model",
                revision="main",
                token=None,
            )

            assert isinstance(loaded_model, BaseFusionModel)
            assert isinstance(loaded_model, DummyFusionModel)

    @mock.patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("Snapshot error"))
    def test_pull_download_failure_raises_deployment_error(self, mock_download):
        with pytest.raises(DeploymentError, match="Failed to download repository"):
            pull_from_hub(repo_id="user/repo")


# ---------------------------------------------------------------------------
# Tests: Missing huggingface_hub ImportError Guard
# ---------------------------------------------------------------------------
class TestImportGuard:
    def test_import_error_message(self):
        """Verify clear ImportError with install instructions appears when
        huggingface_hub is missing."""
        import subprocess

        code = (
            "import sys\n"
            "sys.modules['huggingface_hub'] = None\n"
            "import fusion.integrations.huggingface\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "pip install huggingface_hub transformers" in result.stderr

    def test_import_error_in_process(self):
        """Verify in-process import error when patching sys.modules."""
        import importlib

        saved_mod = sys.modules.get("fusion.integrations.huggingface")
        try:
            with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
                sys.modules.pop("fusion.integrations.huggingface", None)
                with pytest.raises(ImportError, match="pip install huggingface_hub transformers"):
                    importlib.import_module("fusion.integrations.huggingface")
        finally:
            if saved_mod is not None:
                sys.modules["fusion.integrations.huggingface"] = saved_mod


# ---------------------------------------------------------------------------
# Optional Live Test (skipped when HF_TOKEN is not set)
# ---------------------------------------------------------------------------
class TestLiveHub:
    @pytest.mark.skipif(
        "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"].strip(),
        reason="HF_TOKEN environment variable is not set; skipping live Hub test.",
    )
    def test_live_whoami(self):
        import huggingface_hub

        token = os.environ["HF_TOKEN"]
        user_info = huggingface_hub.whoami(token=token)
        assert "name" in user_info
