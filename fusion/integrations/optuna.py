"""
fusion/integrations/optuna.py

Phase 15 — Integrations: Optuna hyperparameter optimization.
Owner: Aman Hingawe

Provides:
    suggest_params_from_space(trial, param_space) -> Dict
    run_optuna_search(model_config, train_loader, val_loader, n_trials,
                      study_name, direction, param_space) -> optuna.Study

Depends on:
    optuna (import-guarded per Phase 15 golden rule)
    Phase 5/11 — model builder + training loop (resilient: injectable
    model_builder_fn / train_and_evaluate_fn overrides let the study run
    in standalone environments and tests; defaults resolve to the full
    FUSION framework when installed).
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, Optional

try:
    import optuna
except ImportError as e:
    raise ImportError(
        "optuna is required for this integration. "
        "Install it with: pip install optuna"
    ) from e

import torch
import torch.nn as nn

try:
    from fusion.exceptions import ConfigError
except ImportError:
    class FusionError(Exception):
        def __init__(self, message, details=None):
            super().__init__(message)
            self.message = message
            self.details = details

    class ConfigError(FusionError):
        pass

try:
    from fusion.utils.logging import get_logger, log_event
    logger = get_logger("integrations.optuna")
except ImportError:
    logger = logging.getLogger("fusion.integrations.optuna")

    def log_event(log, level, event, **fields):
        parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        log.log(level, f"event={event} {parts}".strip())


def suggest_params_from_space(
    trial, param_space: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Suggest one trial's hyperparameters from a typed param space.

    Each entry: {"type": "float"|"int"|"categorical", ...bounds/choices}.
    """
    suggested: Dict[str, Any] = {}
    for name, spec in param_space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ConfigError(
                f"Malformed param space entry for '{name}': expected dict with 'type'",
                details={"name": name, "spec": spec},
            )
        p_type = spec["type"]
        if p_type == "float":
            suggested[name] = trial.suggest_float(
                name, spec.get("low", 0.0), spec.get("high", 1.0),
                log=spec.get("log", False),
            )
        elif p_type == "int":
            suggested[name] = trial.suggest_int(
                name, spec.get("low", 1), spec.get("high", 10),
                log=spec.get("log", False),
            )
        elif p_type == "categorical":
            if "choices" not in spec:
                raise ConfigError(
                    f"Param '{name}' of type categorical requires 'choices'",
                    details={"name": name},
                )
            suggested[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ConfigError(
                f"Unknown param type '{p_type}' for '{name}'",
                details={"supported": ["float", "int", "categorical"]},
            )
    return suggested


def _unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        return batch.get("inputs", batch), batch.get("targets", batch.get("labels"))
    return batch, batch


def _config_to_dict(config) -> Dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def _default_model_builder(model_config):
    try:
        from fusion.models.multimodal.factory import create_multimodal_model
        from fusion.utils.config import Config
    except ImportError as exc:
        raise ConfigError(
            "No model builder available: pass model_builder_fn or "
            "train_and_evaluate_fn to run_optuna_search, or run inside the "
            "full FUSION framework (fusion.models.multimodal.factory).",
            details={"error": str(exc)},
        ) from exc
    cfg = model_config if isinstance(model_config, Config) else Config.from_dict(model_config)
    return create_multimodal_model(cfg)


def _default_train_and_evaluate(
    model_config, train_loader, val_loader, model_builder_fn, epochs
) -> float:
    model = model_builder_fn(model_config)
    lr = float(_config_to_dict(model_config).get("lr", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            inputs, targets = _unpack_batch(batch)
            optimizer.zero_grad()
            out = model(inputs)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = _unpack_batch(batch)
            out = model(inputs)
            logits = out["logits"] if isinstance(out, dict) else out
            total += loss_fn(logits, targets).item() * int(targets.shape[0])
            count += int(targets.shape[0])
    return total / max(count, 1)


def run_optuna_search(
    model_config,
    train_loader,
    val_loader,
    n_trials: int = 10,
    study_name: Optional[str] = None,
    direction: str = "minimize",
    param_space: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    epochs: int = 1,
    model_builder_fn: Optional[Callable] = None,
    train_and_evaluate_fn: Optional[Callable] = None,
) -> "optuna.Study":
    """Run an Optuna study over a FUSION model config.

    The objective suggests params from param_space, merges them into a
    per-trial copy of model_config, builds the model, trains a few epochs,
    evaluates on val_loader, and returns the monitored metric.
    """
    if not param_space:
        raise ConfigError(
            "run_optuna_search requires a non-empty param_space",
            details={"param_space": param_space},
        )
    builder = model_builder_fn or _default_model_builder
    study = optuna.create_study(study_name=study_name, direction=direction)

    def objective(trial) -> float:
        params = suggest_params_from_space(trial, param_space)
        trial_config = copy.deepcopy(_config_to_dict(model_config))
        trial_config.update(params)
        log_event(logger, logging.INFO, "optuna_trial_start",
                  trial=trial.number, params=params)
        if train_and_evaluate_fn is not None:
            metric = train_and_evaluate_fn(trial_config, train_loader, val_loader)
        else:
            metric = _default_train_and_evaluate(
                trial_config, train_loader, val_loader, builder, epochs
            )
        log_event(logger, logging.INFO, "optuna_trial_complete",
                  trial=trial.number, metric=metric)
        return float(metric)

    study.optimize(objective, n_trials=n_trials)
    log_event(logger, logging.INFO, "optuna_study_complete",
              study=study_name, best_value=study.best_value,
              best_params=study.best_params)
    return study


__all__ = ["suggest_params_from_space", "run_optuna_search"]