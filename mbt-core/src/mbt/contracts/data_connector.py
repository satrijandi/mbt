"""Data connector plugin contract - read/write data from warehouses/files."""

from abc import ABC, abstractmethod
from typing import Optional

from mbt.core.data import MBTFrame


class DataConnectorPlugin(ABC):
    """Interface for data source adapters (Snowflake, BigQuery, local files, etc.)."""

    @abstractmethod
    def connect(self, config: dict) -> None:
        """Establish connection to the data source.

        Args:
            config: Connection configuration (credentials, endpoints, etc.)
        """
        ...

    @abstractmethod
    def read_table(
        self,
        table: str,
        columns: Optional[list[str]] = None,
        date_range: Optional[tuple] = None,
    ) -> MBTFrame:
        """Read data from a table.

        Args:
            table: Table name
            columns: Optional list of columns to read (None = all)
            date_range: Optional (start_date, end_date) tuple for filtering

        Returns:
            MBTFrame wrapping the data
        """
        ...

    @abstractmethod
    def write_table(
        self,
        df: MBTFrame,
        table: str,
        mode: str = "overwrite",
    ) -> None:
        """Write predictions/results to a table.

        Args:
            df: Data to write
            table: Target table name
            mode: Write mode (overwrite, append)
        """
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """Test the connection. Called by `mbt debug`."""
        ...
