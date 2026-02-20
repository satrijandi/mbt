"""Split data step - creates train/test splits."""

from typing import Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from mbt.steps.base import Step
from mbt.core.data import PandasFrame
from mbt.utils.temporal import WindowCalculator

logger = logging.getLogger(__name__)


class SplitDataStep(Step):
    """Split data into train and test sets.

    Supports two modes:
    1. Temporal windowing (with data_windows config)
    2. Simple random split (fallback)
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Split data into train and test sets.

        Returns:
            {"train_set": MBTFrame, "test_set": MBTFrame}
        """
        raw_data = inputs["raw_data"]
        df = raw_data.to_pandas()

        # Check for data_windows configuration (supports both full pipeline and step config)
        data_source_config = context.get_config("data_source")
        if data_source_config is None:
            data_source_config = context.get_config("training", "data_source") or {}

        data_windows_config = data_source_config.get("data_windows") if data_source_config else None

        if data_windows_config:
            # Temporal windowing
            train_df, test_df = self._temporal_split(df, data_windows_config, context)
        else:
            # Fallback to simple random split
            train_df, test_df = self._simple_split(df, context)

        logger.info(f"Split completed: {len(train_df)} train rows, {len(test_df)} test rows")
        print(f"  Train set: {len(train_df)} rows")
        print(f"  Test set: {len(test_df)} rows")

        return {
            "train_set": PandasFrame(train_df),
            "test_set": PandasFrame(test_df),
        }

    def _temporal_split(
        self,
        df: pd.DataFrame,
        data_windows_config: dict,
        context: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data using temporal windows."""
        # Get partition key (e.g., snapshot_date)
        schema_config = context.get_config("schema")
        if schema_config is None:
            schema_config = context.get_config("training", "schema") or {}

        partition_key = schema_config.get("identifiers", {}).get("partition_key") if schema_config else None

        if not partition_key:
            raise ValueError("Temporal splitting requires partition_key in schema.identifiers")

        if partition_key not in df.columns:
            raise ValueError(f"Partition key '{partition_key}' not found in data columns")

        # Convert partition_key to datetime
        df = df.copy()
        df[partition_key] = pd.to_datetime(df[partition_key])

        # Get latest available date in data (for label lag handling)
        available_data_end = df[partition_key].max()

        # Calculate windows
        windows = WindowCalculator.calculate_windows(
            execution_date=context.execution_date,
            data_windows_config=data_windows_config,
            available_data_end=available_data_end
        )

        logger.info(f"Temporal windows calculated:")
        logger.info(f"  Train: {windows['train_start'].date()} to {windows['train_end'].date()}")
        logger.info(f"  Test:  {windows['test_start'].date()} to {windows['test_end'].date()}")

        print(f"  Temporal windows:")
        print(f"    Train: {windows['train_start'].date()} to {windows['train_end'].date()}")
        print(f"    Test:  {windows['test_start'].date()} to {windows['test_end'].date()}")

        # Filter data by windows
        train_df = df[
            (df[partition_key] >= windows['train_start']) &
            (df[partition_key] < windows['train_end'])
        ].copy()

        test_df = df[
            (df[partition_key] >= windows['test_start']) &
            (df[partition_key] < windows['test_end'])
        ].copy()

        if len(train_df) == 0:
            raise ValueError(f"No training data in window {windows['train_start']} to {windows['train_end']}")

        if len(test_df) == 0:
            raise ValueError(f"No test data in window {windows['test_start']} to {windows['test_end']}")

        return train_df, test_df

    def _simple_split(self, df: pd.DataFrame, context: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fallback: simple random split."""
        test_size = context.get_config("test_size", default=0.2)
        stratify = context.get_config("stratify", default=True)

        # Get target column (supports both formats)
        target_column = context.get_config("target_column")
        if target_column is None:
            schema_config = context.get_config("schema")
            if schema_config is None:
                schema_config = context.get_config("training", "schema") or {}
            target_column = schema_config.get("target", {}).get("label_column") if schema_config else None

        # Stratified split if requested
        stratify_col = df[target_column] if stratify and target_column and target_column in df else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=42,
        )

        logger.info("Using simple random split (no data_windows configured)")

        return train_df, test_df
