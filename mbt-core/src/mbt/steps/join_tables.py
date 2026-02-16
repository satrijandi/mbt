"""Join tables step - joins multiple tables for training.

Supports:
- Inner, left, right, outer joins
- Multiple tables
- Fan-out checks to prevent accidental data duplication
"""

from typing import Any
import pandas as pd

from mbt.steps.base import Step
from mbt.core.data import PandasFrame


class JoinTablesStep(Step):
    """Join multiple tables into a single training dataset."""

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Join tables based on configuration.

        Returns:
            {"joined_data": MBTFrame} - Joined dataset
        """
        # Get the label table (already loaded)
        label_data = inputs["raw_data"]
        label_df = label_data.to_pandas()

        # Get join configuration
        feature_tables = context.get_config("feature_tables", default=[])

        if not feature_tables:
            # No joins needed, just return label table
            return {"joined_data": label_data}

        print(f"  Joining {len(feature_tables)} feature tables...")

        result_df = label_df.copy()
        initial_rows = len(result_df)

        # Join each feature table
        for i, table_config in enumerate(feature_tables):
            table_name = table_config.get("table")
            join_key = table_config.get("join_key")
            join_type = table_config.get("join_type", "left")
            fan_out_check = table_config.get("fan_out_check", True)

            # Skip if it's the label table itself
            label_table = context.get_config("label_table")
            if table_name == label_table:
                continue

            print(f"    Joining {table_name} on {join_key} ({join_type} join)...")

            # Load feature table
            # TODO: Use data connector to load feature table
            # For Phase 4: Assume it's already in inputs or load from same source
            # This is a simplified implementation
            feature_df = self._load_feature_table(table_name, context)

            # Perform join
            before_rows = len(result_df)
            result_df = result_df.merge(
                feature_df,
                on=join_key,
                how=join_type,
                suffixes=("", f"_{table_name}")
            )
            after_rows = len(result_df)

            # Fan-out check
            if fan_out_check and after_rows > before_rows:
                fanout_ratio = after_rows / before_rows
                raise ValueError(
                    f"Join with {table_name} caused fan-out: "
                    f"{before_rows} → {after_rows} rows ({fanout_ratio:.2f}x). "
                    f"Check for duplicate join keys in {table_name}. "
                    f"Set fan_out_check: false to allow."
                )

            print(f"      {before_rows} rows → {after_rows} rows")

        final_rows = len(result_df)
        print(f"  ✓ Joined {len(feature_tables)} tables: {initial_rows} → {final_rows} rows")

        return {"joined_data": PandasFrame(result_df)}

    def _load_feature_table(self, table_name: str, context: Any) -> pd.DataFrame:
        """Load a feature table.

        Args:
            table_name: Name of the table to load
            context: Runtime context

        Returns:
            DataFrame with feature table data

        Note:
            This is a simplified implementation for Phase 4.
            In a full implementation, this would use the data connector
            to load from the configured data source.
        """
        # For Phase 4: Load from same local directory
        data_path = context.get_config("data_path", default="./sample_data")
        from pathlib import Path

        file_path = Path(data_path) / f"{table_name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Feature table not found: {file_path}\n"
                f"Expected file: {table_name}.csv in {data_path}"
            )

        return pd.read_csv(file_path)
