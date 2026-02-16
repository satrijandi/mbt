"""Data abstraction layer - MBTFrame protocol for lazy evaluation and format negotiation."""

from typing import Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class MBTFrame(Protocol):
    """Protocol for data interchange between steps.

    This allows adapters to implement their own data formats while maintaining
    a common interface. The framework can negotiate formats and avoid unnecessary
    conversions.
    """

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        This is always supported (fallback) but may be expensive for large datasets
        or when converting from native formats like H2OFrame or Spark DataFrame.
        """
        ...

    def num_rows(self) -> int:
        """Return row count without materializing the full dataset."""
        ...

    def columns(self) -> list[str]:
        """Return column names without materializing data."""
        ...

    def schema(self) -> dict[str, str]:
        """Return column name → type mapping without materializing data."""
        ...


class PandasFrame(MBTFrame):
    """Default MBTFrame implementation wrapping pandas DataFrame.

    This is the default format used throughout the framework when no specific
    format is requested.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def num_rows(self) -> int:
        return len(self._df)

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def schema(self) -> dict[str, str]:
        return {col: str(dtype) for col, dtype in self._df.dtypes.items()}

    def __repr__(self) -> str:
        return f"PandasFrame(rows={self.num_rows()}, cols={len(self.columns())})"
