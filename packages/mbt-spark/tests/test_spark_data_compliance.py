"""The Spark adapter against the shared data-seam compliance suite (A-1).

Tagged e2e like the rest of the Spark suite: it needs a real local session.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mbt_spark.data import SparkDataAdapter

from mbt_adapter_base.compliance import DataAdapterCompliance
from mbt_adapter_base.specs import SourceTable

pytestmark = [pytest.mark.e2e]


class TestSparkDataAdapterCompliance(DataAdapterCompliance):
    def make_adapter(self, root: Path, rows: pa.Table) -> tuple[Any, Any]:
        data = root / "data"
        data.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows, data / "rows.parquet")
        return (
            SparkDataAdapter({"master": "local[2]", "root": str(root)}),
            SourceTable(name="rows", path="data/*.parquet"),
        )
