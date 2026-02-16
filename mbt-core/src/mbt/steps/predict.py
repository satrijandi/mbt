"""Predict step - generates predictions using trained model.

Generates:
- Class predictions
- Probability predictions (for classification)
"""

from typing import Any
import pandas as pd
import numpy as np

from mbt.steps.base import Step
from mbt.core.data import PandasFrame
from mbt.core.registry import PluginRegistry


class PredictStep(Step):
    """Generate predictions using trained model."""

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Generate predictions.

        Returns:
            {
                "predictions": array of predictions,
                "prediction_probabilities": array of probabilities (classification only)
            }
        """
        # Get model and transformed data
        model = inputs["model"]
        transformed_data = inputs["transformed_data"]

        df = transformed_data.to_pandas()

        print(f"  Generating predictions for {len(df)} rows...")

        # Get framework to use its predict method
        framework_name = context.get_config("framework", default="sklearn")
        problem_type = context.get_config("problem_type", default="binary_classification")

        registry = PluginRegistry()

        try:
            framework = registry.get("mbt.frameworks", framework_name)
        except Exception as e:
            print(f"    ⚠ Warning: Cannot load framework plugin, using model.predict() directly")
            # Fallback to direct model prediction
            predictions = model.predict(df)

            # Try to get probabilities
            try:
                prediction_probs = model.predict_proba(df)
            except Exception:
                prediction_probs = None

            return {
                "predictions": predictions,
                "prediction_probabilities": prediction_probs,
            }

        # Use framework plugin
        try:
            # Wrap data as MBTFrame
            data_frame = PandasFrame(df)

            # Get predictions
            predictions = framework.predict(model, data_frame)

            print(f"    ✓ Generated {len(predictions)} predictions")

            # Get probabilities for classification
            prediction_probs = None
            if problem_type in ["binary_classification", "multiclass_classification"]:
                try:
                    prediction_probs = framework.predict_proba(model, data_frame)
                    print(f"    ✓ Generated probability predictions")
                except Exception as e:
                    print(f"    ⚠ Could not generate probabilities: {e}")

            return {
                "predictions": predictions,
                "prediction_probabilities": prediction_probs,
            }

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
