"""Tests for the optional MLflow integration."""

import importlib
import sys
from types import ModuleType
from unittest import mock

import pytest


def _install_fake_mlflow():
    """Install a lightweight fake mlflow module for isolated tests."""
    fake = ModuleType("mlflow")
    fake.set_tracking_uri = mock.Mock()
    fake.set_experiment = mock.Mock()
    fake.start_run = mock.Mock()
    fake.end_run = mock.Mock()
    fake.log_params = mock.Mock()
    fake.log_metrics = mock.Mock()
    fake.log_artifact = mock.Mock()
    fake.log_artifacts = mock.Mock()
    sys.modules["mlflow"] = fake
    return fake


class DummyModel:
    """Minimal model whose save() writes into a directory."""

    def save(self, path):
        import os

        target = os.path.join(path, "weights.pt") if os.path.isdir(path) else path
        with open(target, "w") as f:
            f.write("weights")


@pytest.fixture
def fake_mlflow():
    """Provide a fake mlflow module and reload the integration."""
    saved = sys.modules.get("mlflow")
    fake = _install_fake_mlflow()

    sys.modules.pop("fusion.integrations.mlflow", None)
    module = importlib.import_module("fusion.integrations.mlflow")

    yield fake, module

    sys.modules.pop("fusion.integrations.mlflow", None)
    if saved is not None:
        sys.modules["mlflow"] = saved
    else:
        sys.modules.pop("mlflow", None)


class TestMLflowLogger:
    """Tests for MLflowLogger behavior."""

    def test_init_sets_uri_experiment_and_starts_run(self, fake_mlflow):
        mlflow, module = fake_mlflow

        module.MLflowLogger(tracking_uri="file:./mlruns", experiment_name="exp")

        mlflow.set_tracking_uri.assert_called_once_with("file:./mlruns")
        mlflow.set_experiment.assert_called_once_with("exp")
        mlflow.start_run.assert_called_once()

    def test_log_params(self, fake_mlflow):
        mlflow, module = fake_mlflow
        lg = module.MLflowLogger()

        lg.log_params({"lr": 0.001})

        mlflow.log_params.assert_called_once_with({"lr": 0.001})

    def test_log_metrics(self, fake_mlflow):
        mlflow, module = fake_mlflow
        lg = module.MLflowLogger()

        lg.log_metrics({"loss": 0.5}, step=3)

        mlflow.log_metrics.assert_called_once_with({"loss": 0.5}, step=3)

    def test_log_model_logs_saved_artifact(self, fake_mlflow):
        mlflow, module = fake_mlflow
        lg = module.MLflowLogger()

        lg.log_model(DummyModel(), artifact_path="model")

        mlflow.log_artifacts.assert_called_once()
        assert mlflow.log_artifacts.call_args.kwargs["artifact_path"] == "model"

    def test_finish_ends_run(self, fake_mlflow):
        mlflow, module = fake_mlflow
        module.MLflowLogger().finish()
        mlflow.end_run.assert_called_once_with()


class TestImportGuard:
    """Tests for the optional dependency import guard."""

    def test_import_error_message(self):
        saved_module = sys.modules.get("fusion.integrations.mlflow")

        try:
            sys.modules.pop("fusion.integrations.mlflow", None)

            with mock.patch.dict(sys.modules, {"mlflow": None}):
                with pytest.raises(ImportError, match="pip install mlflow"):
                    importlib.import_module("fusion.integrations.mlflow")
        finally:
            sys.modules.pop("fusion.integrations.mlflow", None)

            if saved_module is not None:
                sys.modules["fusion.integrations.mlflow"] = saved_module
