"""DuckDB data connector for MBT.

Reads and writes data from DuckDB databases (file-based or in-memory).
"""

from typing import Optional

from mbt.contracts.data_connector import DataConnectorPlugin
from mbt.core.data import PandasFrame, MBTFrame


class DuckDBConnector(DataConnectorPlugin):
    """DuckDB data connector.

    Configuration (in profiles.yaml):
        data_connector:
          type: duckdb
          config:
            path: ./warehouse.duckdb  # or :memory: for in-memory
            schema: main              # optional, default: main
            read_only: false          # optional, default: false
    """

    def __init__(self):
        self._conn = None
        self._config: dict = {}

    def connect(self, config: dict) -> None:
        """Open DuckDB connection from config."""
        import duckdb

        path = config.get("path", ":memory:")
        read_only = config.get("read_only", False)

        self._conn = duckdb.connect(database=path, read_only=read_only)
        self._config = config

    def read_table(
        self,
        table: str,
        columns: Optional[list[str]] = None,
        date_range: Optional[tuple] = None,
    ) -> MBTFrame:
        """Read data from DuckDB table."""
        if self._conn is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        schema = self._config.get("schema", "main")
        col_clause = ", ".join(columns) if columns else "*"
        query = f'SELECT {col_clause} FROM "{schema}"."{table}"'

        df = self._conn.execute(query).fetchdf()
        return PandasFrame(df)

    def write_table(
        self,
        df: MBTFrame,
        table: str,
        mode: str = "overwrite",
    ) -> None:
        """Write data to DuckDB table."""
        if self._conn is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        schema = self._config.get("schema", "main")
        pandas_df = df.to_pandas()

        # DuckDB can query pandas DataFrames directly by variable name
        if mode == "overwrite":
            self._conn.execute(
                f'CREATE OR REPLACE TABLE "{schema}"."{table}" AS SELECT * FROM pandas_df'
            )
        else:
            self._conn.execute(
                f'INSERT INTO "{schema}"."{table}" SELECT * FROM pandas_df'
            )

    def validate_connection(self) -> bool:
        """Test the DuckDB connection."""
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False
