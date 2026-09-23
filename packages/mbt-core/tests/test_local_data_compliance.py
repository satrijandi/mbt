"""The local DuckDB adapter against the shared data-seam compliance suite (A-1).

The data seam had no compliance suite until v5: every dataset test ran real
DuckDB against the one adapter, and the three adapters agreed on the build
recipe only because each had copied it.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mbt.adapters.local.data import LocalDataAdapter
from mbt.contracts import SourceTable
from mbt_adapter_base.compliance import DataAdapterCompliance


class TestLocalDataAdapterCompliance(DataAdapterCompliance):
    def make_adapter(self, root: Path, rows: pa.Table) -> tuple[Any, Any]:
        data = root / "data"
        data.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows, data / "rows.parquet")
        return (
            LocalDataAdapter({"root": str(root)}),
            SourceTable(name="rows", path="data/*.parquet"),
        )
