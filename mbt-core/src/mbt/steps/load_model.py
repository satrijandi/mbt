"""Load model step - loads trained model and artifacts from model registry.

Loads:
- Trained model
- Preprocessing artifacts (scaler, encoder, feature_selector)
- Training metadata
"""

from typing import Any
from mbt.steps.base import Step
from mbt.core.registry import PluginRegistry


class LoadModelStep(Step):
    """Load trained model and artifacts from model registry.

    Used in serving pipelines to load a previously trained model.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Load model and artifacts.

        Returns:
            {
                "model": trained model,
                "scaler": normalization scaler (if exists),
                "encoder": categorical encoder (if exists),
                "feature_selector": feature selector info (if exists),
                "model_metadata": run metadata
            }
        """
        # Get model source configuration
        registry_name = context.get_config("model_registry", default="mlflow")
        run_id = context.get_config("model_run_id")

        if not run_id:
            raise ValueError(
                "model_run_id not specified in serving configuration. "
                "Pass via: --vars run_id=<mlflow_run_id>"
            )

        print(f"  Loading model from {registry_name} run: {run_id}")

        # Load model registry plugin
        plugin_registry = PluginRegistry()

        try:
            model_registry = plugin_registry.get("mbt.model_registries", registry_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model registry '{registry_name}': {e}\n"
                f"Install with: pip install mbt-{registry_name}"
            )

        # Load model
        print(f"    Loading model...")
        try:
            model = model_registry.load_model(run_id)
            print(f"      ✓ Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from run {run_id}: {e}")

        # Load artifacts
        print(f"    Loading artifacts...")
        try:
            artifacts = model_registry.load_artifacts(run_id)
            print(f"      ✓ Loaded {len(artifacts)} artifacts: {list(artifacts.keys())}")
        except Exception as e:
            print(f"      ⚠ Warning: Failed to load artifacts: {e}")
            artifacts = {}

        # Get run metadata
        try:
            metadata = model_registry.get_run_info(run_id)
        except Exception:
            metadata = {}

        # Build result dictionary with all artifacts (None if not present)
        result = {
            "model": model,
            "model_metadata": metadata,
            "scaler": artifacts.get("scaler"),
            "encoder": artifacts.get("encoder"),
            "feature_selector": artifacts.get("feature_selector"),
        }

        # Report which artifacts are available
        available_artifacts = []
        if result["scaler"] is not None:
            available_artifacts.append("scaler")
        if result["encoder"] is not None:
            available_artifacts.append("encoder")
        if result["feature_selector"] is not None:
            available_artifacts.append("feature_selector")

        if available_artifacts:
            print(f"      ✓ Artifacts available: {', '.join(available_artifacts)}")
        else:
            print(f"      ⚠ No transformation artifacts (scaler, encoder, feature_selector)")

        print(f"  ✓ Model and artifacts loaded successfully")

        return result
