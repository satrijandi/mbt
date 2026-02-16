"""Log run step - logs training run to model registry (e.g., MLflow)."""

from typing import Any

from mbt.steps.base import Step
from mbt.core.registry import PluginRegistry


class LogRunStep(Step):
    """Log training run to model registry.

    Collects all metrics, parameters, artifacts, and tags from the training
    pipeline and logs them to the configured model registry (e.g., MLflow).

    This enables:
    - Experiment tracking and comparison
    - Model versioning
    - Artifact management
    - Serving pipeline model loading
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Log training run to model registry.

        Args:
            inputs: Dictionary containing:
                - model: Trained model object
                - train_metrics: Training metrics dict
                - eval_metrics: Evaluation metrics dict
                - Other artifacts (scaler, encoder, etc.)
            context: Runtime context with pipeline configuration

        Returns:
            {"run_id": str} - Model registry run ID
        """
        # Get model registry configuration
        registry_name = context.get_config("registry", default="mlflow")

        # Load model registry plugin
        plugin_registry = PluginRegistry()

        try:
            model_registry = plugin_registry.get("mbt.model_registries", registry_name)
        except Exception as e:
            print(f"  Warning: Model registry '{registry_name}' not available: {e}")
            print(f"  Skipping run logging. Install with: pip install mbt-{registry_name}")
            return {"run_id": None}

        # Get pipeline name from context
        pipeline_name = context.get_config("pipeline_name", default="unknown")

        # Collect metrics
        metrics = {}

        if "train_metrics" in inputs:
            metrics.update(inputs["train_metrics"])

        if "eval_metrics" in inputs:
            metrics.update(inputs["eval_metrics"])

        # Collect parameters
        params = {}

        # Framework configuration
        framework_name = context.get_config("framework", default="sklearn")
        params["framework"] = framework_name

        framework_config = context.get_config("framework_config", default={})
        for key, value in framework_config.items():
            params[f"framework.{key}"] = value

        # Problem type
        problem_type = context.get_config("problem_type", default="unknown")
        params["problem_type"] = problem_type

        # Target column
        target_column = context.get_config("target_column", default="unknown")
        params["target_column"] = target_column

        # Collect artifacts
        artifacts = {}

        # Always include the model
        if "model" in inputs:
            artifacts["model"] = inputs["model"]

        # Include other artifacts (scaler, encoder, feature_selector, etc.)
        for key, value in inputs.items():
            if key not in ["train_metrics", "eval_metrics", "model"]:
                # Check if it looks like an artifact (not a dict/list)
                if value is not None and not isinstance(value, (dict, list)):
                    artifacts[key] = value

        # Collect tags
        tags = {
            "framework": framework_name,
            "problem_type": problem_type,
        }

        # Add custom tags from pipeline configuration
        custom_tags = context.get_config("tags", default={})
        if isinstance(custom_tags, dict):
            tags.update(custom_tags)

        # Log run to model registry
        print(f"  Logging run to {registry_name}...")
        print(f"    Metrics: {list(metrics.keys())}")
        print(f"    Artifacts: {list(artifacts.keys())}")

        try:
            run_id = model_registry.log_run(
                pipeline_name=pipeline_name,
                metrics=metrics,
                params=params,
                artifacts=artifacts,
                tags=tags,
            )

            print(f"  ✓ Logged to {registry_name}: run_id = {run_id}")

            return {"run_id": run_id}
        except Exception as e:
            print(f"  ✗ Failed to log run: {e}")
            return {"run_id": None}
