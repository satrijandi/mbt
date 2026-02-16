"""Normalize step - applies feature scaling.

Supported methods:
- standard_scaler: Standardize features (mean=0, std=1)
- min_max_scaler: Scale to range [0, 1]
- robust_scaler: Scale using median and IQR (robust to outliers)
"""

from typing import Any
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from mbt.steps.base import Step
from mbt.core.data import PandasFrame


class NormalizeStep(Step):
    """Normalize numeric features using specified scaler.

    Fits scaler on training data, transforms both train and test.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Normalize features.

        Returns:
            {
                "normalized_train": MBTFrame,
                "normalized_test": MBTFrame,
                "scaler": fitted scaler object
            }
        """
        train_set = inputs["train_set"]
        test_set = inputs["test_set"]

        train_df = train_set.to_pandas()
        test_df = test_set.to_pandas()

        # Get configuration
        method = context.get_config("normalization_method", default="standard_scaler")
        target_column = context.get_config("target_column")
        schema_config = context.get_config("schema", default={})

        # Build list of columns to exclude
        columns_to_exclude = [target_column]

        # Add identifier columns
        if "identifiers" in schema_config:
            if "primary_key" in schema_config["identifiers"]:
                columns_to_exclude.append(schema_config["identifiers"]["primary_key"])
            if "partition_key" in schema_config["identifiers"]:
                pk = schema_config["identifiers"]["partition_key"]
                if pk:
                    columns_to_exclude.append(pk)

        # Add ignored columns
        if "ignored_columns" in schema_config:
            columns_to_exclude.extend(schema_config["ignored_columns"])

        # Remove duplicates
        columns_to_exclude = list(set(columns_to_exclude))

        # Select numeric columns only
        numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
        numeric_cols = [col for col in numeric_cols if col not in columns_to_exclude]

        print(f"  Normalizing {len(numeric_cols)} numeric features with {method}...")

        # Select scaler
        if method == "standard_scaler":
            scaler = StandardScaler()
        elif method == "min_max_scaler":
            scaler = MinMaxScaler()
        elif method == "robust_scaler":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Fit on training data
        scaler.fit(train_df[numeric_cols])

        # Transform both train and test
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()

        train_normalized[numeric_cols] = scaler.transform(train_df[numeric_cols])
        test_normalized[numeric_cols] = scaler.transform(test_df[numeric_cols])

        print(f"    Features: {numeric_cols[:3]}... ({len(numeric_cols)} total)")

        return {
            "normalized_train": PandasFrame(train_normalized),
            "normalized_test": PandasFrame(test_normalized),
            "scaler": scaler,
        }
