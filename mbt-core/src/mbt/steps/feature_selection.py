"""Feature selection step - selects most important features.

Supported methods:
- variance_threshold: Remove low-variance features
- correlation: Remove highly correlated features
- mutual_info: Select features by mutual information with target
- lgbm_importance: Select features by LightGBM importance
"""

from typing import Any
import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

from mbt.steps.base import Step
from mbt.core.data import PandasFrame

logger = logging.getLogger(__name__)


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

            elif method_name == "lgbm_importance":
                threshold = method_config.get("threshold", 0.95)
                kept = self._lgbm_importance(X_train, y_train, threshold, problem_type, selected_features)
                removed = selected_features - kept
                selected_features = kept
                print(f"    ✓ LGBM importance: kept {len(kept)} features (threshold={threshold}), removed {len(removed)}")

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
            "train_set": PandasFrame(selected_train),
            "test_set": PandasFrame(selected_test),
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

    def _lgbm_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float,
        problem_type: str,
        selected: set
    ) -> set:
        """Select features using LightGBM importance with cross-validation.

        Uses stratified K-fold cross-validation to calculate robust feature importances,
        then selects features based on either:
        - Zero importance filtering (removes features with 0 importance)
        - Cumulative importance threshold (e.g., 0.95 for top 95% important features)

        Args:
            X: Feature dataframe
            y: Target series
            threshold: If >= 1.0, removes zero-importance features only.
                      If < 1.0, uses cumulative importance threshold (e.g., 0.95).
            problem_type: Problem type (classification or regression)
            selected: Set of currently selected features

        Returns:
            Set of selected features
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import StratifiedKFold, KFold
        except ImportError:
            logger.error("lightgbm not installed - skipping lgbm_importance. Install with: pip install lightgbm")
            print("    ⚠ lightgbm not installed - skipping lgbm_importance")
            return selected

        # Only consider currently selected features
        X_selected = X[[col for col in X.columns if col in selected]]

        if len(X_selected.columns) == 0:
            return selected

        # Calculate scale_pos_weight for imbalanced classification
        scale_pos_weight = 1
        if problem_type in ["binary_classification", "multiclass_classification"]:
            value_counts = y.value_counts()
            if len(value_counts) == 2:
                scale_pos_weight = int(value_counts.iloc[0] / value_counts.iloc[1])

        # Initialize feature importances array
        feature_importances = np.zeros(X_selected.shape[1])

        # Setup cross-validation
        n_splits = 5
        if problem_type in ["binary_classification", "multiclass_classification"]:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Cross-validation to get robust feature importances
        logger.info(f"Running {n_splits}-fold CV for feature importance calculation...")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_selected, y)):
            X_train_fold = X_selected.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X_selected.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # Create model
            if problem_type in ["binary_classification", "multiclass_classification"]:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    boosting_type='gbdt',
                    n_estimators=100,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbose=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )

            try:
                # Train with early stopping
                if problem_type in ["binary_classification", "multiclass_classification"]:
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                    )
                else:
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                    )

                # Accumulate feature importances
                feature_importances += model.feature_importances_ / n_splits

            except Exception as e:
                logger.warning(f"Fold {fold_idx + 1} failed: {e}, using zero importance")
                continue

        # Create feature importances DataFrame
        importances_df = pd.DataFrame({
            'feature': X_selected.columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        # Select features based on threshold
        if threshold >= 1.0:
            # Zero-importance filtering: remove features with exactly 0 importance
            zero_features = importances_df[importances_df['importance'] == 0.0]['feature'].tolist()
            selected_features = set(importances_df[importances_df['importance'] > 0.0]['feature'].tolist())

            logger.info(f"LGBM zero-importance filter: removed {len(zero_features)} features with 0 importance")
            print(f"    ✓ LGBM: removed {len(zero_features)} zero-importance features")

        else:
            # Cumulative importance threshold
            importances_df['importance_normalized'] = (
                importances_df['importance'] / importances_df['importance'].sum()
            )
            importances_df['cumulative_importance'] = importances_df['importance_normalized'].cumsum()

            # Find features needed for threshold
            selected_df = importances_df[importances_df['cumulative_importance'] <= threshold]

            # If no features selected, keep at least the top feature
            if len(selected_df) == 0:
                selected_df = importances_df.head(1)
                logger.warning(f"LGBM threshold {threshold} too strict, keeping top feature only")

            selected_features = set(selected_df['feature'].tolist())

            logger.info(
                f"LGBM cumulative importance: selected {len(selected_features)}/{len(X_selected.columns)} features "
                f"for {threshold*100}% importance"
            )

        # Ensure at least one feature is selected
        if len(selected_features) == 0:
            selected_features = set([importances_df.iloc[0]['feature']])
            logger.warning("No features selected by LGBM, keeping top feature")

        return selected_features
