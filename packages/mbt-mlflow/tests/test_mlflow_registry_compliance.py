"""The MLflow registry against the shared champion-record contract (A-5).

The round-trip case here is what ``promote.py``, ``oot_check.py`` and
``runners.py`` each separately assumed about a registered version's tags, and
what no single test asserted before v5.
"""

from pathlib import Path
from typing import Any

from mbt_mlflow.adapter import MlflowRegistry

from mbt_adapter_base.compliance import RegistryAdapterCompliance


class TestMlflowRegistryCompliance(RegistryAdapterCompliance):
    def make_registry(self, root: Path) -> Any:
        return MlflowRegistry({"uri": f"sqlite:///{root / 'mlflow.db'}"})
