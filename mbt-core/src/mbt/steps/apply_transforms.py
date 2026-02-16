"""Apply transforms step - applies preprocessing from training.

Applies saved transformations to scoring data:
- Normalization (scaler)
- Categorical encoding (encoder)
- Feature selection
"""

from typing import Any
import pandas as pd
from mbt.steps.base import Step
from mbt.core.data import PandasFrame


class ApplyTransformsStep(Step):
    """Apply preprocessing transformations to scoring data.

    Applies the same transformations that were used during training,
    ensuring schema consistency between training and serving.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Apply transformations.

        Returns:
            {"transformed_data": MBTFrame} - Transformed scoring data
        """
        # Get scoring data
        scoring_data = inputs["scoring_data"]
        df = scoring_data.to_pandas()

        print(f"  Applying transformations to {len(df)} rows...")

        # Get schema for column filtering
        schema_config = context.get_config("schema", default=None)

        # Filter out non-feature columns first (identifiers, target, ignored)
        if schema_config:
            columns_to_exclude = []

            # Add identifier columns
            identifiers = schema_config.get("identifiers", {})
            if "primary_key" in identifiers:
                columns_to_exclude.append(identifiers["primary_key"])
            if "partition_key" in identifiers:
                columns_to_exclude.append(identifiers["partition_key"])

            # Add target column (shouldn't be in scoring data, but be safe)
            target = schema_config.get("target", {})
            if "label_column" in target:
                columns_to_exclude.append(target["label_column"])

            # Add ignored columns
            ignored = schema_config.get("ignored_columns", [])
            columns_to_exclude.extend(ignored)

            # Filter out these columns if they exist
            columns_to_keep = [col for col in df.columns if col not in columns_to_exclude]
            df = df[columns_to_keep]

            if columns_to_exclude:
                print(f"    Filtered out columns: {[col for col in columns_to_exclude if col in scoring_data.to_pandas().columns]}")

        # Track which transforms are applied
        transforms_applied = []

        # Apply feature selection first (if exists)
        if "feature_selector" in inputs and inputs["feature_selector"] is not None:
            df = self._apply_feature_selection(df, inputs["feature_selector"])
            transforms_applied.append("feature_selection")

        # Apply scaler (if exists)
        if "scaler" in inputs and inputs["scaler"] is not None:
            df = self._apply_scaler(df, inputs["scaler"])
            transforms_applied.append("normalization")

        # Apply encoder (if exists)
        if "encoder" in inputs and inputs["encoder"] is not None:
            df = self._apply_encoder(df, inputs["encoder"])
            transforms_applied.append("encoding")

        if transforms_applied:
            print(f"    ✓ Applied: {', '.join(transforms_applied)}")
        else:
            print(f"    No transformations to apply")

        return {"transformed_data": PandasFrame(df)}

    def _apply_feature_selection(self, df: pd.DataFrame, selector_info: dict) -> pd.DataFrame:
        """Apply feature selection.

        Args:
            df: Input dataframe
            selector_info: Feature selector info dict with 'selected_features'

        Returns:
            DataFrame with only selected features
        """
        selected_features = selector_info.get("selected_features", [])

        if not selected_features:
            print(f"      ⚠ Warning: No selected features in selector info")
            return df

        # Keep only selected features that exist in df
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]

        if missing_features:
            print(f"      ⚠ Warning: {len(missing_features)} features missing in scoring data")

        # Also keep identifier columns (if they exist)
        identifier_cols = [col for col in df.columns if col not in selected_features]
        # Filter to only actual identifier columns (customer_id, etc.)
        identifier_cols = [col for col in identifier_cols if col in df.columns[:5]]  # Heuristic

        result_cols = available_features + identifier_cols
        return df[result_cols]

    def _apply_scaler(self, df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        """Apply normalization scaler.

        Args:
            df: Input dataframe
            scaler: Fitted sklearn scaler

        Returns:
            DataFrame with normalized numeric features
        """
        # Get numeric columns that were scaled during training
        # Scaler has feature_names_in_ attribute (sklearn >= 1.0)
        if hasattr(scaler, "feature_names_in_"):
            numeric_cols = list(scaler.feature_names_in_)
        elif hasattr(scaler, "n_features_in_"):
            # Fallback: use first n numeric columns
            all_numeric = df.select_dtypes(include=["int64", "float64"]).columns
            numeric_cols = list(all_numeric[:scaler.n_features_in_])
        else:
            print(f"      ⚠ Warning: Cannot determine scaler features, skipping")
            return df

        # Check if columns exist
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        if missing_cols:
            print(f"      ⚠ Warning: Missing columns for scaling: {missing_cols}")
            numeric_cols = [col for col in numeric_cols if col in df.columns]

        if not numeric_cols:
            return df

        # Apply transformation
        result = df.copy()
        result[numeric_cols] = scaler.transform(df[numeric_cols])

        return result

    def _apply_encoder(self, df: pd.DataFrame, encoder: Any) -> pd.DataFrame:
        """Apply categorical encoder.

        Args:
            df: Input dataframe
            encoder: Fitted encoder (dict of LabelEncoders or None for one-hot)

        Returns:
            DataFrame with encoded categorical features
        """
        if encoder is None:
            # One-hot encoding was used during training
            # This is complex to apply in serving without saving column info
            print(f"      ⚠ Warning: One-hot encoding in serving requires column mapping (not implemented)")
            return df

        # Label encoding
        if isinstance(encoder, dict):
            result = df.copy()

            for col, le in encoder.items():
                if col not in df.columns:
                    print(f"      ⚠ Warning: Column '{col}' not found for encoding")
                    continue

                # Handle unseen categories
                result[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

            return result

        print(f"      ⚠ Warning: Unknown encoder type: {type(encoder)}")
        return df
