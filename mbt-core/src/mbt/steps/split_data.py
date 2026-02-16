"""Split data step - creates train/test splits."""

from typing import Any
from sklearn.model_selection import train_test_split

from mbt.steps.base import Step
from mbt.core.data import PandasFrame


class SplitDataStep(Step):
    """Split data into train and test sets.

    For Phase 1: Simple train/test split with optional stratification.
    For Phase 4: Temporal windowing based on data_windows configuration.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Split data into train and test sets.

        Returns:
            {"train_set": MBTFrame, "test_set": MBTFrame}
        """
        raw_data = inputs["raw_data"]
        df = raw_data.to_pandas()

        # Get configuration
        test_size = context.get_config("test_size", default=0.2)
        stratify = context.get_config("stratify", default=True)
        target_column = context.get_config("target_column")

        # Stratified split if requested
        stratify_col = df[target_column] if stratify and target_column else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=42,
        )

        print(f"  Train set: {len(train_df)} rows")
        print(f"  Test set: {len(test_df)} rows")

        return {
            "train_set": PandasFrame(train_df),
            "test_set": PandasFrame(test_df),
        }
