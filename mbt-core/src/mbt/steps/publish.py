"""Publish step - writes predictions to output destination.

Writes:
- Predictions
- Probabilities (for classification)
- Identifiers (customer_id, etc.)
- Metadata (execution_date, model_run_id)
"""

from typing import Any
import pandas as pd
from datetime import datetime
from pathlib import Path

from mbt.steps.base import Step
from mbt.core.registry import PluginRegistry
from mbt.core.data import PandasFrame


class PublishStep(Step):
    """Publish predictions to output destination."""

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Write predictions.

        Returns:
            {"output_path": str} - Path or destination of written predictions
        """
        # Get predictions and original data
        predictions = inputs["predictions"]
        prediction_probs = inputs.get("prediction_probabilities")
        scoring_data = inputs["scoring_data"]

        df = scoring_data.to_pandas()

        # Get configuration
        output_config = context.get_config("output_config", default={})
        include_probabilities = output_config.get("include_probabilities", True)
        destination = output_config.get("destination", "local_file")

        # Get metadata
        run_id = context.get_config("run_id", default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model_run_id = context.get_config("model_run_id", default="unknown")

        print(f"  Publishing {len(predictions)} predictions...")

        # Build output dataframe
        # For database destination, only include identifier columns (not features)
        if destination == "database":
            # Get primary key from schema config
            schema = context.get_config("schema", default={})
            pk = schema.get("identifiers", {}).get("primary_key", None)
            if pk and pk in df.columns:
                output_df = df[[pk]].copy()
            else:
                # Fall back to just the first column as identifier
                output_df = df[[df.columns[0]]].copy()
        else:
            output_df = df.copy()

        # Add predictions
        output_df["prediction"] = predictions

        # Add probabilities for classification
        if prediction_probs is not None and include_probabilities:
            if prediction_probs.ndim == 1:
                # Binary classification with single probability
                output_df["prediction_probability"] = prediction_probs
            elif prediction_probs.shape[1] == 2:
                # Binary classification with [p0, p1]
                output_df["prediction_probability"] = prediction_probs[:, 1]
            else:
                # Multiclass - add all class probabilities
                for i in range(prediction_probs.shape[1]):
                    output_df[f"prediction_probability_class_{i}"] = prediction_probs[:, i]

        # Add metadata
        output_df["execution_date"] = datetime.now().isoformat()
        output_df["model_run_id"] = model_run_id
        output_df["serving_run_id"] = run_id

        if destination == "database":
            # Write to database via data connector plugin
            table_name = output_config.get("table", "predictions")

            registry = PluginRegistry()
            dc_config = context.profile_config.get("data_connector", {})
            dc_type = dc_config.get("type", "local_file")
            connector = registry.get("mbt.data_connectors", dc_type)
            connector.connect(dc_config.get("config", {}))

            connector.write_table(PandasFrame(output_df), table_name, mode="append")

            print(f"    Wrote {len(output_df)} rows to table: {table_name}")
            return {"output_path": f"database://{table_name}"}
        else:
            # Write to CSV file (default)
            output_path = output_config.get("path", "./predictions.csv")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_df.to_csv(output_path, index=False)

            print(f"    Wrote predictions to: {output_path}")
            print(f"    Columns: {list(output_df.columns)}")

            return {"output_path": str(output_path)}
