"""Feature selection step - selects most important features.

Supported methods:
- variance_threshold: Remove low-variance features
- correlation: Remove highly correlated features
- mutual_info: Select features by mutual information with target
"""

from typing import Any
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

from mbt.steps.base import Step
from mbt.core.data import PandasFrame


class FeatureSelectionStep(Step):
    """Select most important features.

    Fits selector on training data, applies to both train and test.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Select features.

        Returns:
            {
                "selected_train": MBTFrame,
                "selected_test": MBTFrame,
                "feature_selector": selector info dict
            }
        """
        # Get most recent version of data (after encoding)
        train_set = inputs.get("encoded_train") or inputs.get("normalized_train") or inputs.get("train_set")
        test_set = inputs.get("encoded_test") or inputs.get("normalized_test") or inputs.get("test_set")

        train_df = train_set.to_pandas()
        test_df = test_set.to_pandas()

        # Get configuration
        methods = context.get_config("feature_selection_methods", default=[])
        target_column = context.get_config("target_column")
        problem_type = context.get_config("problem_type")
        schema_config = context.get_config("schema", default={})

        if not methods:
            print("  Feature selection not enabled")
            return {
                "selected_train": train_set,
                "selected_test": test_set,
                "feature_selector": None,
            }

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

        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in columns_to_exclude]
        X_train = train_df[feature_cols]
        y_train = train_df[target_column]

        print(f"  Selecting features from {len(feature_cols)} total features...")

        selected_features = set(feature_cols)

        # Apply each selection method
        for method_config in methods:
            method_name = method_config.get("name")

            if method_name == "variance_threshold":
                threshold = method_config.get("threshold", 0.0)
                removed = self._variance_threshold(X_train, threshold, selected_features)
                print(f"    ✓ Variance threshold: removed {len(removed)} features")

            elif method_name == "correlation":
                threshold = method_config.get("threshold", 0.95)
                removed = self._correlation_filter(X_train, threshold, selected_features)
                print(f"    ✓ Correlation filter: removed {len(removed)} features")

            elif method_name == "mutual_info":
                k = method_config.get("k", 10)
                kept = self._mutual_info_selection(X_train, y_train, k, problem_type, selected_features)
                removed = selected_features - kept
                selected_features = kept
                print(f"    ✓ Mutual info: kept top {k} features, removed {len(removed)}")

            else:
                print(f"    ⚠ Unknown method: {method_name}")

        selected_features = sorted(selected_features)

        print(f"  ✓ Selected {len(selected_features)}/{len(feature_cols)} features")

        # Apply selection
        selected_train = train_df[selected_features + columns_to_exclude].copy()
        selected_test = test_df[selected_features + columns_to_exclude].copy()

        selector_info = {
            "selected_features": selected_features,
            "original_count": len(feature_cols),
            "selected_count": len(selected_features),
        }

        return {
            "selected_train": PandasFrame(selected_train),
            "selected_test": PandasFrame(selected_test),
            "feature_selector": selector_info,
        }

    def _variance_threshold(self, X: pd.DataFrame, threshold: float, selected: set) -> set:
        """Remove low-variance features.

        Args:
            X: Feature dataframe
            threshold: Variance threshold
            selected: Set of currently selected features (modified in place)

        Returns:
            Set of removed features
        """
        # Only apply to numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col in selected]

        if len(numeric_cols) == 0:
            return set()

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X[numeric_cols])

        # Get features to remove
        removed = set()
        for i, col in enumerate(numeric_cols):
            if not selector.get_support()[i]:
                removed.add(col)
                selected.discard(col)

        return removed

    def _correlation_filter(self, X: pd.DataFrame, threshold: float, selected: set) -> set:
        """Remove highly correlated features.

        Args:
            X: Feature dataframe
            threshold: Correlation threshold (0-1)
            selected: Set of currently selected features (modified in place)

        Returns:
            Set of removed features
        """
        # Only apply to numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col in selected]

        if len(numeric_cols) < 2:
            return set()

        # Compute correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()

        # Find pairs above threshold
        removed = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > threshold:
                    # Remove the second feature
                    col_to_remove = corr_matrix.columns[j]
                    if col_to_remove not in removed:
                        removed.add(col_to_remove)
                        selected.discard(col_to_remove)

        return removed

    def _mutual_info_selection(
        self, X: pd.DataFrame, y: pd.Series, k: int, problem_type: str, selected: set
    ) -> set:
        """Select top k features by mutual information.

        Args:
            X: Feature dataframe
            y: Target series
            k: Number of features to keep
            problem_type: Problem type (classification or regression)
            selected: Set of currently selected features

        Returns:
            Set of selected features
        """
        # Only consider currently selected features
        X_selected = X[[col for col in X.columns if col in selected]]

        # Compute mutual information
        if problem_type in ["binary_classification", "multiclass_classification"]:
            mi_scores = mutual_info_classif(X_selected, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_selected, y, random_state=42)

        # Select top k
        top_k_indices = np.argsort(mi_scores)[-k:]
        top_k_features = X_selected.columns[top_k_indices].tolist()

        return set(top_k_features)
