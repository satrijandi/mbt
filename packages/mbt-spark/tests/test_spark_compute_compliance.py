"""The Spark compute adapter declares what it implements (FEEDBACK v7 C-1).

Tagged e2e like the rest of the Spark suite.
"""

from pathlib import Path
from typing import Any

import pytest
from mbt_spark.compute import SparkComputeAdapter

from mbt_adapter_base.compliance import ComputeAdapterCompliance

pytestmark = [pytest.mark.e2e]


class TestSparkComputeCompliance(ComputeAdapterCompliance):
    def make_compute(self, root: Path) -> Any:
        return SparkComputeAdapter({"master": "local[1]"})
