"""
tests/integrations/test_optuna.py

Unit tests for ``fusion.integrations.optuna``.

Tests cover:
    - suggest_params_from_space handles float / int / categorical entries
    - suggest_params_from_space rejects malformed specs and unknown types
    - run_optuna_search runs n_trials and returns a study with best_params
    - run_optuna_search honours the direction argument
    - run_optuna_search rejects an empty param_space
    - ImportError message when optuna is missing

Run with:
    pytest tests/integrations/test_optuna.py -v
"""

from __future__ import annotations

import subprocess
import sys

import optuna
import pytest

from fusion.integrations.optuna import (
    run_optuna_search,
    suggest_params_from_space,
)

PARAM_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "layers": {"type": "int", "low": 1, "high": 3},
    "pool": {"type": "categorical", "choices": ["mean", "max"]},
}


class TestSuggestParamsFromSpace:
    def test_suggests_all_keys(self):
        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params_from_space(trial, PARAM_SPACE)
        assert set(params) == {"lr", "layers", "pool"}
        assert 1e-4 <= params["lr"] <= 1e-1
        assert params["layers"] in (1, 2, 3)
        assert params["pool"] in ("mean", "max")

    def test_malformed_spec_raises(self):
        from fusion.integrations.optuna import ConfigError

        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ConfigError):
            suggest_params_from_space(trial, {"lr": {"low": 1e-4}})
        with pytest.raises(ConfigError):
            suggest_params_from_space(trial, {"lr": 0.01})

    def test_unknown_type_raises(self):
        from fusion.integrations.optuna import ConfigError

        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ConfigError):
            suggest_params_from_space(trial, {"x": {"type": "bool"}})


class TestRunOptunaSearch:
    def test_runs_n_trials_and_returns_study(self):
        study = run_optuna_search(
            model_config={"lr": 1e-3},
            train_loader=None,
            val_loader=None,
            n_trials=4,
            study_name="unit",
            direction="minimize",
            param_space=PARAM_SPACE,
            train_and_evaluate_fn=lambda cfg, tr, va: float(cfg["layers"]) + cfg["lr"],
        )
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 4
        assert set(study.best_params) == {"lr", "layers", "pool"}

    def test_maximize_direction(self):
        study = run_optuna_search(
            model_config={},
            train_loader=None,
            val_loader=None,
            n_trials=3,
            direction="maximize",
            param_space={"layers": {"type": "int", "low": 1, "high": 5}},
            train_and_evaluate_fn=lambda cfg, tr, va: float(cfg["layers"]),
        )
        assert study.best_value == max(t.value for t in study.trials)

    def test_empty_param_space_raises(self):
        from fusion.integrations.optuna import ConfigError

        with pytest.raises(ConfigError):
            run_optuna_search(
                model_config={},
                train_loader=None,
                val_loader=None,
                param_space={},
            )


class TestImportGuard:
    def test_import_error_message(self):
        code = (
            "import sys\n"
            "sys.modules['optuna'] = None\n"
            "import fusion.integrations.optuna\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "pip install optuna" in result.stderr
        