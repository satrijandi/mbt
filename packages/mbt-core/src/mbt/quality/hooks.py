"""hooks.py loading and invocation (TSD §5.8, FR-RES-07).

Hooks run inside the training job only, never in the coordinator. The hook
file's bytes are hashed into the node's config_hash at compile time, so
editing a hook marks the model ``state:modified``.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa

from mbt.contracts import HookContext
from mbt.exceptions import AdapterError

HOOK_TRANSFORM = "transform_features"
HOOK_METRICS = "custom_metrics"


class ModelHooks:
    """A loaded hooks module for one model."""

    def __init__(self, module: Any, path: Path) -> None:
        self._module = module
        self.path = path
        self.has_transform = callable(getattr(module, HOOK_TRANSFORM, None))
        self.has_custom_metrics = callable(getattr(module, HOOK_METRICS, None))

    def transform_features(self, table: pa.Table, ctx: HookContext) -> pa.Table:
        if not self.has_transform:
            return table
        result = getattr(self._module, HOOK_TRANSFORM)(table, ctx)
        if not isinstance(result, pa.Table):
            raise AdapterError(
                f"hooks {HOOK_TRANSFORM} must return a pyarrow.Table, got {type(result).__name__}",
                path=self.path,
            )
        return result

    def custom_metrics(self, predictions: pa.Table, ctx: HookContext) -> dict[str, float]:
        if not self.has_custom_metrics:
            return {}
        result = getattr(self._module, HOOK_METRICS)(predictions, ctx)
        if not isinstance(result, dict):
            raise AdapterError(
                f"hooks {HOOK_METRICS} must return a dict of metric name -> float",
                path=self.path,
            )
        return {str(k): float(v) for k, v in result.items()}


def load_hooks(project_dir: Path, hooks_path: str) -> ModelHooks:
    path = project_dir / hooks_path
    if not path.is_file():
        raise AdapterError(
            f"hooks file not found: {path}",
            hint="the manifest pinned a hooks file that no longer exists",
        )
    module_name = f"_mbt_hooks_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guards
        raise AdapterError(f"cannot import hooks module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise AdapterError(f"hooks module failed to import: {exc!r}", path=path) from exc
    return ModelHooks(module, path)
