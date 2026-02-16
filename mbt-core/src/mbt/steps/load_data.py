"""Load data step - reads data from configured data source."""

from typing import Any

from mbt.steps.base import Step
from mbt.builtins.local_connector import LocalFileConnector


class LoadDataStep(Step):
    """Load data from data source.

    For Phase 1: Uses LocalFileConnector to read CSV files.
    For Phase 3+: Uses data_connector plugin from profiles.yaml
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

        # Initialize connector (Phase 1: hardcoded local)
        connector = LocalFileConnector()
        connector.connect({"data_path": "./sample_data"})

        # Read data
        raw_data = connector.read_table(label_table)

        print(f"  Loaded {raw_data.num_rows()} rows, {len(raw_data.columns())} columns")

        return {"raw_data": raw_data}
