"""The fake registry against the shared champion-record contract (A-5).

Two adapters make a seam real, and the registry seam had no compliance suite at
all: ``mbt-mlflow/tests/`` was the de facto contract, so this fake existed to
make core's tests run rather than to prove the abstraction (C-5). Running both
against one suite is what turns it into a contract.
"""

from pathlib import Path
from typing import Any

from mbt_testing import FakeRegistryAdapter

from mbt_adapter_base.compliance import RegistryAdapterCompliance


class TestFakeRegistryCompliance(RegistryAdapterCompliance):
    def make_registry(self, root: Path) -> Any:
        return FakeRegistryAdapter({"root": str(root / "registry")})
