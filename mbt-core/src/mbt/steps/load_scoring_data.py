"""Load scoring data step - loads data for prediction.

Similar to load_data but for serving pipeline.
Loads data from scoring_table configured in serving section.
"""

from typing import Any

from mbt.steps.base import Step
from mbt.core.registry import PluginRegistry


class LoadScoringDataStep(Step):
    """Load scoring data for prediction."""

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Load scoring data.

        Returns:
            {"scoring_data": MBTFrame} - Raw scoring data
        """
        # Get scoring table configuration
        scoring_table = context.get_config("scoring_table")

        print(f"  Loading scoring data from: {scoring_table}")

        # Resolve data connector from profile config
        registry = PluginRegistry()
        dc_config = context.profile_config.get("data_connector", {})
        dc_type = dc_config.get("type", "local_file")
        connector = registry.get("mbt.data_connectors", dc_type)
        connector.connect(dc_config.get("config", {"data_path": "./sample_data"}))

        scoring_data = connector.read_table(scoring_table)

        print(f"    Loaded {scoring_data.num_rows()} rows, {len(scoring_data.columns())} columns")
        print(f"    Columns: {scoring_data.columns()}")

        return {"scoring_data": scoring_data}
