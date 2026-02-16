"""Load scoring data step - loads data for prediction.

Similar to load_data but for serving pipeline.
Loads data from scoring_table configured in serving section.
"""

from typing import Any
import pandas as pd
from pathlib import Path

from mbt.steps.base import Step
from mbt.core.data import PandasFrame


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

        # For Phase 5: Hardcoded local file loading
        # In production: would use data connector plugin
        data_path = Path("sample_data") / f"{scoring_table}.csv"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Scoring data not found: {data_path}\n"
                f"Expected CSV file at: {data_path}"
            )

        # Load CSV
        df = pd.read_csv(data_path)

        print(f"    ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns)}")

        return {"scoring_data": PandasFrame(df)}
