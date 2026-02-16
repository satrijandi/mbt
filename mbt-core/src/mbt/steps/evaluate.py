"""Evaluate model step - computes metrics on test set."""

from typing import Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from mbt.steps.base import Step


class EvaluateStep(Step):
    """Evaluate trained model on test set.

    Computes metrics appropriate for the problem type.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Evaluate model on test set.

        Returns:
            {"eval_metrics": dict, "eval_plots": dict}
        """
        model = inputs["model"]
        test_set = inputs["test_set"]
        test_df = test_set.to_pandas()

        # Get configuration
        target_column = context.get_config("target_column")
        problem_type = context.get_config("problem_type")
        primary_metric = context.get_config("primary_metric", default="roc_auc")
        additional_metrics = context.get_config("additional_metrics", default=[])
        schema_config = context.get_config("schema", default={})

        # Build list of columns to exclude (same as training step)
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

        # Separate features and target
        X_test = test_df.drop(columns=columns_to_exclude, errors='ignore')
        y_test = test_df[target_column]

        # Generate predictions using framework plugin
        # First, try to get the framework from context to use its predict method
        # If not available, fall back to model.predict() for sklearn compatibility
        framework_name = context.get_config("framework", default="sklearn")

        try:
            from mbt.core.registry import PluginRegistry
            from mbt.core.data import PandasFrame

            registry = PluginRegistry()
            framework = registry.get("mbt.frameworks", framework_name)

            X_test_frame = PandasFrame(X_test)
            y_pred = framework.predict(model, X_test_frame)
        except Exception:
            # Fallback for Phase 1 compatibility
            y_pred = model.predict(X_test)

        # Compute metrics based on problem type
        metrics = {}

        if problem_type in ["binary_classification", "multiclass_classification"]:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)

            # For binary classification, compute additional metrics
            if problem_type == "binary_classification":
                if "precision" in additional_metrics or "precision" == primary_metric:
                    metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)

                if "recall" in additional_metrics or "recall" == primary_metric:
                    metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)

                if "f1" in additional_metrics or "f1" == primary_metric:
                    metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)

                if "roc_auc" in additional_metrics or "roc_auc" == primary_metric:
                    # ROC AUC requires predict_proba
                    try:
                        # Try framework plugin first
                        y_proba = framework.predict_proba(model, X_test_frame)
                        # For binary classification, take positive class probability
                        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                            y_proba = y_proba[:, 1]
                        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                    except Exception:
                        # Fallback to model.predict_proba for sklearn compatibility
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)[:, 1]
                            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

        # Print metrics
        print(f"  Evaluation metrics:")
        for name, value in metrics.items():
            print(f"    {name}: {value:.4f}")

        return {
            "eval_metrics": metrics,
            "eval_plots": {},  # Phase 1: no plots yet
        }
