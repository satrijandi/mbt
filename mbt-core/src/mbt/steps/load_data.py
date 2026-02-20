"""Load data step - reads data from configured data source."""

from typing import Any
import pandas as pd
import logging

from mbt.steps.base import Step
from mbt.core.registry import PluginRegistry
from mbt.core.data import PandasFrame

logger = logging.getLogger(__name__)


class LoadDataStep(Step):
    """Load data from data source.

    Supports:
    - Single table loading (label_table only)
    - Multi-table joins (label_table + feature_tables)

    Uses data_connector plugin resolved from profiles.yaml.
    Falls back to local_file connector if no profile config.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Load data from source.

        Returns:
            {"raw_data": MBTFrame}
        """
        # Get configuration (supports both full pipeline and step config)
        data_source_config = context.get_config("data_source")
        if data_source_config is None:
            # Fallback to legacy config path
            data_source_config = context.get_config("training", "data_source")

        if data_source_config is None:
            raise ValueError("data_source config not found")

        label_table = data_source_config.get("label_table")
        if not label_table:
            raise ValueError("label_table not specified in data_source config")

        # Resolve data connector from profile config
        registry = PluginRegistry()
        dc_config = context.profile_config.get("data_connector", {})
        dc_type = dc_config.get("type", "local_file")
        connector = registry.get("mbt.data_connectors", dc_type)
        connector.connect(dc_config.get("config", {"data_path": "./sample_data"}))

        # Load label table
        logger.info(f"Loading label table: {label_table}")
        raw_data = connector.read_table(label_table)
        df = raw_data.to_pandas()

        logger.info(f"Loaded label table '{label_table}': {len(df)} rows, {len(df.columns)} columns")
        print(f"  Loaded label table: {len(df)} rows, {len(df.columns)} columns")

        # Load and join feature tables if configured
        feature_tables = data_source_config.get("feature_tables")
        if feature_tables:
            df = self._join_feature_tables(df, feature_tables, connector, context)

        return {"raw_data": PandasFrame(df)}

    def _join_feature_tables(
        self,
        label_df: pd.DataFrame,
        feature_tables: list,
        connector: Any,
        context: Any
    ) -> pd.DataFrame:
        """Join multiple feature tables with label table.

        Args:
            label_df: Label table DataFrame
            feature_tables: List of feature table configurations
            connector: Data connector instance
            context: Run context

        Returns:
            Combined DataFrame with all features joined
        """
        result_df = label_df.copy()
        initial_rows = len(result_df)

        logger.info(f"Joining {len(feature_tables)} feature table(s)")
        print(f"  Joining {len(feature_tables)} feature table(s)...")

        for ft_config in feature_tables:
            table_name = ft_config.get("table")
            join_key = ft_config.get("join_key", ["customer_id", "snapshot_date"])
            join_type = ft_config.get("join_type", "left")
            fan_out_check = ft_config.get("fan_out_check", True)

            # Load feature table
            logger.info(f"Loading feature table: {table_name}")
            feature_data = connector.read_table(table_name)
            feature_df = feature_data.to_pandas()

            logger.info(f"Loaded feature table '{table_name}': {len(feature_df)} rows, {len(feature_df.columns)} columns")
            print(f"    - {table_name}: {len(feature_df)} rows, {len(feature_df.columns)} columns")

            # Perform join
            result_df = result_df.merge(
                feature_df,
                on=join_key,
                how=join_type,
                suffixes=("", f"_{table_name}")
            )

            # Fan-out detection
            if fan_out_check and len(result_df) > initial_rows * 1.1:
                raise ValueError(
                    f"Join with '{table_name}' caused unexpected row increase: "
                    f"{initial_rows} → {len(result_df)} rows. "
                    f"Check for duplicate keys in join columns {join_key}. "
                    f"Set 'fan_out_check: false' to disable this check."
                )

            logger.info(f"After joining '{table_name}': {len(result_df)} rows, {len(result_df.columns)} columns")

        print(f"  Final joined data: {len(result_df)} rows, {len(result_df.columns)} columns")

        return result_df
