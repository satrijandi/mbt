"""Train model step - trains ML model using configured framework.

Phase 2: Uses framework plugin from registry for modular framework support.
"""

from typing import Any
import pandas as pd

from mbt.steps.base import Step
from mbt.core.data import PandasFrame
from mbt.core.registry import PluginRegistry


class TrainModelStep(Step):
    """Train ML model using framework plugin.

    Supports any framework with an installed adapter (sklearn, h2o_automl, etc.).
    The framework is loaded dynamically from the plugin registry.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Train model on training data.

        Returns:
            {"model": trained_model, "train_metrics": dict}
        """
        train_set = inputs["train_set"]
        train_df = train_set.to_pandas()

        # Get configuration
        target_column = context.get_config("target_column")
        problem_type = context.get_config("problem_type")
        framework_name = context.get_config("framework", default="sklearn")
        framework_config = context.get_config("framework_config", default={})
        schema_config = context.get_config("schema", default={})

        # Build list of columns to exclude (target + identifiers + ignored)
        columns_to_exclude = [target_column]

        # Add identifier columns
        if "identifiers" in schema_config:
            if "primary_key" in schema_config["identifiers"]:
                columns_to_exclude.append(schema_config["identifiers"]["primary_key"])
            if "partition_key" in schema_config["identifiers"]:
                pk = schema_config["identifiers"]["partition_key"]
                if pk:  # Only add if not None
                    columns_to_exclude.append(pk)

        # Add ignored columns
        if "ignored_columns" in schema_config:
            columns_to_exclude.extend(schema_config["ignored_columns"])

        # Remove duplicates
        columns_to_exclude = list(set(columns_to_exclude))

        # Separate features and target
        X_train_df = train_df.drop(columns=columns_to_exclude, errors='ignore')
        y_train_df = train_df[[target_column]]  # Keep as DataFrame

        print(f"  Training {problem_type} model with {framework_name}")
        print(f"  Features: {list(X_train_df.columns)[:5]}... ({len(X_train_df.columns)} total)")

        # Phase 2: Use framework plugin from registry
        registry = PluginRegistry()
        framework = registry.get("mbt.frameworks", framework_name)

        # Setup framework (e.g., h2o.init())
        framework.setup({})

        try:
            # Wrap data as MBTFrames
            X_train = PandasFrame(X_train_df)
            y_train = PandasFrame(y_train_df)

            # Train model via plugin
            model = framework.train(X_train, y_train, framework_config)

            # Get training metrics from framework
            train_metrics = framework.get_training_metrics(model)

            # If no metrics available, compute basic accuracy
            if not train_metrics:
                try:
                    predictions = framework.predict(model, X_train)
                    y_true = y_train_df.iloc[:, 0].values
                    accuracy = (predictions == y_true).mean()
                    train_metrics = {"train_accuracy": accuracy}
                    print(f"  Training accuracy: {accuracy:.4f}")
                except Exception:
                    train_metrics = {}

            return {
                "model": model,
                "train_metrics": train_metrics,
            }
        finally:
            # Teardown framework (e.g., h2o.cluster().shutdown())
            framework.teardown()
