"""Snowflake data connector for MBT.

Reads and writes data from Snowflake warehouses using snowflake-connector-python.
"""

from typing import Optional

from mbt.contracts.data_connector import DataConnectorPlugin
from mbt.core.data import PandasFrame, MBTFrame


class SnowflakeConnector(DataConnectorPlugin):
    """Snowflake data connector.

    Configuration (in profiles.yaml):
        data_connector:
          type: snowflake
          config:
            account: my_account
            user: my_user
            password: my_password
            warehouse: COMPUTE_WH
            database: MY_DB
            schema: PUBLIC
            role: MY_ROLE  # optional
    """

    def __init__(self):
        self._conn = None
        self._config: dict = {}

    def connect(self, config: dict) -> None:
        """Create Snowflake connection from config."""
        import snowflake.connector

        connect_params = {
            "account": config.get("account", ""),
            "user": config.get("user", ""),
            "password": config.get("password", ""),
            "warehouse": config.get("warehouse", ""),
            "database": config.get("database", ""),
            "schema": config.get("schema", "PUBLIC"),
        }

        role = config.get("role")
        if role:
            connect_params["role"] = role

        self._conn = snowflake.connector.connect(**connect_params)
        self._config = config

    def read_table(
        self,
        table: str,
        columns: Optional[list[str]] = None,
        date_range: Optional[tuple] = None,
    ) -> MBTFrame:
        """Read data from Snowflake table."""
        if self._conn is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        schema = self._config.get("schema", "PUBLIC")
        col_clause = ", ".join(columns) if columns else "*"
        query = f'SELECT {col_clause} FROM "{schema}"."{table}"'

        cur = self._conn.cursor()
        try:
            cur.execute(query)
            df = cur.fetch_pandas_all()
        finally:
            cur.close()

        return PandasFrame(df)

    def write_table(
        self,
        df: MBTFrame,
        table: str,
        mode: str = "overwrite",
    ) -> None:
        """Write data to Snowflake table."""
        from snowflake.connector.pandas_tools import write_pandas

        if self._conn is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        schema = self._config.get("schema", "PUBLIC")
        pandas_df = df.to_pandas()

        if mode == "overwrite":
            cur = self._conn.cursor()
            try:
                cur.execute(f'TRUNCATE TABLE IF EXISTS "{schema}"."{table}"')
            finally:
                cur.close()

        write_pandas(
            self._conn,
            pandas_df,
            table_name=table,
            schema=schema,
            auto_create_table=True,
        )

    def validate_connection(self) -> bool:
        """Test the Snowflake connection."""
        if self._conn is None:
            return False
        try:
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT 1")
            finally:
                cur.close()
            return True
        except Exception:
            return False
