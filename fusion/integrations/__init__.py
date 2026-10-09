# integrations package
from __future__ import annotations

import importlib
from typing import Any, Optional


def require(module_name: str, pip_name: Optional[str] = None) -> Any:
    """Import and return an optional dependency, or raise a helpful ImportError.

    Args:
        module_name: Name of the Python module/package to import (e.g. 'transformers').
        pip_name: Name of the pip package to install if different (e.g. 'huggingface_hub transformers').
            If not specified, defaults to *module_name*.

    Returns:
        The imported module.

    Raises:
        ImportError: Explaining how to install the missing package.
    """
    pkg = pip_name if pip_name is not None else module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"'{module_name}' is required for this integration. "
            f"Install it with: pip install {pkg}"
        ) from exc


__all__ = ["require"]
