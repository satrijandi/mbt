"""Local file connector - reads CSV/Parquet files from local filesystem."""

from pathlib import Path
from typing import Optional
import pandas as pd

from mbt.contracts.data_connector import DataConnectorPlugin
from mbt.core.data import PandasFrame, MBTFrame


class LocalFileConnector(DataConnectorPlugin):
    """Reads data from local CSV/Parquet files."""

    def __init__(self):
        self.data_path: Optional[Path] = None

    def connect(self, config: dict) -> None:
        """Set data path from config."""
        self.data_path = Path(config.get("data_path", "./sample_data"))
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

    def read_table(
        self,
        table: str,
        columns: Optional[list[str]] = None,
        date_range: Optional[tuple] = None,
    ) -> MBTFrame:
        """Read CSV or Parquet file.

        Table name is used as filename (table.csv or table.parquet).
        """
        if self.data_path is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        # Try CSV first, then Parquet
        csv_path = self.data_path / f"{table}.csv"
        parquet_path = self.data_path / f"{table}.parquet"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Table not found: {table} (tried .csv and .parquet)")

        # Apply column filtering if requested
        if columns:
            df = df[columns]

        # Date range filtering (Phase 4 feature - stub for now)
        if date_range:
            # TODO: Implement temporal filtering
            pass

        return PandasFrame(df)

    def write_table(
        self,
        df: MBTFrame,
        table: str,
        mode: str = "overwrite",
    ) -> None:
        """Write data to CSV file."""
        if self.data_path is None:
            raise RuntimeError("Connector not connected. Call connect() first.")

        output_path = self.data_path / f"{table}.csv"

        pandas_df = df.to_pandas()

        if mode == "overwrite" or not output_path.exists():
            pandas_df.to_csv(output_path, index=False)
        elif mode == "append":
            pandas_df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            raise ValueError(f"Invalid write mode: {mode}")

    def validate_connection(self) -> bool:
        """Check if data directory exists."""
        return self.data_path is not None and self.data_path.exists()
