"""The local compute adapter declares what it implements (FEEDBACK v7 C-1)."""

from pathlib import Path
from typing import Any

from mbt.adapters.local.compute import LocalComputeAdapter
from mbt_adapter_base.capabilities import Capability, supports
from mbt_adapter_base.compliance import ComputeAdapterCompliance


class TestLocalComputeCompliance(ComputeAdapterCompliance):
    def make_compute(self, root: Path) -> Any:
        return LocalComputeAdapter({})


def test_local_compute_declares_the_kill_seam() -> None:
    """``--fail-fast`` reclaims in-flight jobs only through ``terminate``."""
    assert supports(LocalComputeAdapter({}), Capability.TERMINATE)
