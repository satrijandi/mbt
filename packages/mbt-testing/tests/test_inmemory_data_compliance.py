"""The in-memory DataAdapter against the shared data-seam compliance suite.

Two adapters make a seam real (A-1). This is the second one for the data seam,
and it is what proves ``build_dataset_materialization`` holds the recipe rather
than the adapters holding copies of it: this engine shares no code with DuckDB,
Spark or Snowflake, and produces the same splits, the same events and the same
severities.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
from mbt_testing import InMemoryDataAdapter

from mbt_adapter_base.compliance import DataAdapterCompliance
from mbt_adapter_base.specs import SourceTable


class TestInMemoryDataAdapterCompliance(DataAdapterCompliance):
    def make_adapter(self, root: Path, rows: pa.Table) -> tuple[Any, Any]:
        source = SourceTable(name="rows", identifier="rows")
        return InMemoryDataAdapter({"root": str(root), "tables": {"rows": rows}}), source
