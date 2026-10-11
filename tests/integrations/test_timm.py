"""
tests/integrations/test_timm.py

Unit tests for ``fusion.integrations.timm``.

Tests cover:
    - list_timm_models returns sorted names and respects the filter
    - load_timm_encoder builds a num_classes=0 feature extractor with the
      correct output dimension and a working encode() path
    - load_timm_encoder raises EncoderError for an unknown model name
    - ImportError message when timm is missing

Run with:
    pytest tests/integrations/test_timm.py -v
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from fusion.integrations.timm import list_timm_models, load_timm_encoder


class TestListTimmModels:
    def test_returns_sorted_list(self):
        models = list_timm_models()
        assert isinstance(models, list)
        assert models == sorted(models)
        assert len(models) > 0

    def test_filter_respected(self):
        models = list_timm_models("resnet18")
        assert "resnet18" in models
        assert all("resnet18" in m for m in models)


class TestLoadTimmEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        # pretrained=False keeps the test offline and fast.
        return load_timm_encoder("resnet18", pretrained=False)

    def test_output_dim(self, encoder):
        assert encoder.get_output_dim() == 512

    def test_encode_shape(self, encoder):
        out = encoder.encode(torch.randn(2, 3, 224, 224))
        tensor = out.data if hasattr(out, "data") else out
        assert tuple(tensor.shape) == (2, 512)

    def test_unknown_model_raises(self):
        from fusion.integrations.timm import EncoderError

        with pytest.raises(EncoderError):
            load_timm_encoder("not_a_real_timm_model_xyz", pretrained=False)


class TestImportGuard:
    def test_import_error_message(self):
        code = (
            "import sys\n"
            "sys.modules['timm'] = None\n"
            "import fusion.integrations.timm\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "pip install timm" in result.stderr
        