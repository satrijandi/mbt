"""PostgreSQL data connector for MBT.

Reads and writes data from PostgreSQL databases using SQLAlchemy.
"""

from typing import Optional

from mbt.contracts.data_connector import DataConnectorPlugin
from mbt.core.data import PandasFrame, MBTFrame


class PostgresConnector(DataConnectorPlugin):
    """PostgreSQL data connector.

    Configuration (in profiles.yaml):
        data_connector:
          type: postgres
          config:
            host: localhost
            port: 5432
            database: warehouse
            schema: public
            user: mbt_user
            password: mbt_password
    """

    def __init__(self):
        self._engine = None
        self._config: dict = {}

    def connect(self, config: dict) -> None:
        """Create SQLAlchemy engine from config."""
        from sqlalchemy import create_engine

        host = config.get("host", "localhost")
        port = config.get("port", 5432)
        database = config.get("database", "warehouse")
        user = config.get("user", "mbt_user")
        password = config.get("password", "")

        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self._engine = create_engine(url)
        self._config = config

    def read_table(
        self,
        table: str,
        columns: Optional[list[str]] = None,
        date_range: Optional[tuple] = None,
    ) -> MBTFrame:
        """Read data from PostgreSQL table."""
        import pandas as pd

        if self._engine is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        schema = self._config.get("schema", "public")
        col_clause = ", ".join(columns) if columns else "*"
        query = f'SELECT {col_clause} FROM "{schema}"."{table}"'

        df = pd.read_sql(query, self._engine)
        return PandasFrame(df)

    def write_table(
        self,
        df: MBTFrame,
        table: str,
        mode: str = "overwrite",
    ) -> None:
        """Write data to PostgreSQL table."""
        if self._engine is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        schema = self._config.get("schema", "public")
        pandas_df = df.to_pandas()
        if_exists = "replace" if mode == "overwrite" else "append"

        pandas_df.to_sql(
            table, self._engine, schema=schema, if_exists=if_exists, index=False
        )

    def validate_connection(self) -> bool:
        """Test the PostgreSQL connection."""
        if self._engine is None:
            return False
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
