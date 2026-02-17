"""Load data step - reads data from configured data source."""

from typing import Any

from mbt.steps.base import Step
from mbt.core.registry import PluginRegistry


class LoadDataStep(Step):
    """Load data from data source.

    Uses data_connector plugin resolved from profiles.yaml.
    Falls back to local_file connector if no profile config.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Load data from source.

        Returns:
            {"raw_data": MBTFrame}
        """
        # Get configuration
        label_table = context.get_config("label_table")

        if not label_table:
            raise ValueError("label_table not specified in data_source config")

        # Resolve data connector from profile config
        registry = PluginRegistry()
        dc_config = context.profile_config.get("data_connector", {})
        dc_type = dc_config.get("type", "local_file")
        connector = registry.get("mbt.data_connectors", dc_type)
        connector.connect(dc_config.get("config", {"data_path": "./sample_data"}))

        # Read data
        raw_data = connector.read_table(label_table)

        print(f"  Loaded {raw_data.num_rows()} rows, {len(raw_data.columns())} columns")

        return {"raw_data": raw_data}
