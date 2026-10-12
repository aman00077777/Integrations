"""Tests for the optional TensorBoard integration."""

import importlib
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensorboard")

import torch

from fusion.integrations.tensorboard import TensorBoardLogger


def test_logs_write_event_files(tmp_path):
    """Verify that scalars, embeddings, and images produce event files."""
    logger = TensorBoardLogger(log_dir=str(tmp_path))

    try:
        logger.log_scalars("train/loss", 0.5, step=1)
        logger.log_embedding(
            torch.randn(4, 8),
            metadata=["a", "b", "c", "d"],
            step=1,
        )
        logger.log_images(
            "imgs",
            torch.rand(2, 3, 8, 8),
            step=1,
        )
        logger.flush()
    finally:
        logger.close()

    assert list(tmp_path.rglob("events.out.tfevents.*"))


def test_import_guard(monkeypatch):
    """Verify that a missing optional dependency raises a clear ImportError."""
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    sys.modules.pop("fusion.integrations.tensorboard", None)

    try:
        with pytest.raises(ImportError, match="pip install tensorboard"):
            importlib.import_module("fusion.integrations.tensorboard")
    finally:
        sys.modules.pop("fusion.integrations.tensorboard", None)