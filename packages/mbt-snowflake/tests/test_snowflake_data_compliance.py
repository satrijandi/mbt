"""The Snowflake adapter against the shared data-seam compliance suite (A-1).

The stub runs the adapter's real generated SQL in DuckDB, so the bucket-edge
case here genuinely pins Snowflake's ``MD5_NUMBER_LOWER64`` expression to the
one cross-adapter reference rather than to a copy of it.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
from mbt_snowflake.adapter import SnowflakeDataAdapter
from snowflake_stub_helpers import FakeSourceTable, StubConnection

from mbt_adapter_base.compliance import DataAdapterCompliance


class TestSnowflakeDataAdapterCompliance(DataAdapterCompliance):
    def make_adapter(self, root: Path, rows: pa.Table) -> tuple[Any, Any]:
        # Unquoted Snowflake identifiers arrive UPPERCASE; the adapter
        # normalizes back to mbt's lowercase spec conventions.
        upper = rows.rename_columns([c.upper() for c in rows.column_names])
        stub = StubConnection(tables={"ANALYTICS.GOLD.ROWS": upper})
        adapter = SnowflakeDataAdapter({"database": "ANALYTICS", "schema": "GOLD"})
        adapter._connection = stub  # type: ignore[assignment]
        return adapter, FakeSourceTable(name="rows", identifier="ROWS")
