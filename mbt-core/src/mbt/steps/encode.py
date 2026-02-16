"""Encode step - encodes categorical features.

Supported methods:
- one_hot: One-hot encoding (creates dummy variables)
- label: Label encoding (maps categories to integers)
"""

from typing import Any
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mbt.steps.base import Step
from mbt.core.data import PandasFrame


class EncodeStep(Step):
    """Encode categorical features.

    Fits encoder on training data, transforms both train and test.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Encode categorical features.

        Returns:
            {
                "encoded_train": MBTFrame,
                "encoded_test": MBTFrame,
                "encoder": fitted encoder object (dict of LabelEncoders)
            }
        """
        train_set = inputs.get("normalized_train") or inputs.get("train_set")
        test_set = inputs.get("normalized_test") or inputs.get("test_set")

        train_df = train_set.to_pandas()
        test_df = test_set.to_pandas()

        # Get configuration
        method = context.get_config("encoding_method", default="one_hot")
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

        # Select categorical columns
        categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns
        categorical_cols = [col for col in categorical_cols if col not in columns_to_exclude]

        if len(categorical_cols) == 0:
            print("  No categorical features to encode")
            return {
                "encoded_train": train_set,
                "encoded_test": test_set,
                "encoder": None,
            }

        print(f"  Encoding {len(categorical_cols)} categorical features with {method}...")

        if method == "one_hot":
            # One-hot encoding
            train_encoded = pd.get_dummies(
                train_df,
                columns=categorical_cols,
                drop_first=True,  # Avoid multicollinearity
                prefix=categorical_cols,
            )

            test_encoded = pd.get_dummies(
                test_df,
                columns=categorical_cols,
                drop_first=True,
                prefix=categorical_cols,
            )

            # Align columns (test may have different categories)
            test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

            encoder = None  # No encoder object for one-hot

        elif method == "label":
            # Label encoding
            train_encoded = train_df.copy()
            test_encoded = test_df.copy()

            encoders = {}

            for col in categorical_cols:
                le = LabelEncoder()
                train_encoded[col] = le.fit_transform(train_df[col].astype(str))

                # Handle unseen categories in test
                test_col = test_df[col].astype(str)
                test_encoded[col] = test_col.map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

                encoders[col] = le

            encoder = encoders

        else:
            raise ValueError(f"Unknown encoding method: {method}")

        print(f"    Features: {categorical_cols[:3]}... ({len(categorical_cols)} total)")

        return {
            "encoded_train": PandasFrame(train_encoded),
            "encoded_test": PandasFrame(test_encoded),
            "encoder": encoder,
        }
