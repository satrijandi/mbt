"""MLflow model registry adapter for MBT.

Provides experiment tracking, model logging, and artifact management via MLflow.
"""

import mlflow
import mlflow.pyfunc
from typing import Any
from pathlib import Path
import pickle
import tempfile
import shutil


# Import ModelRegistryPlugin from mbt-core
try:
    from mbt.contracts.model_registry import ModelRegistryPlugin
except ImportError:
    # Fallback for development/testing
    class ModelRegistryPlugin:
        """Stub for development."""
        pass


class MLflowRegistry(ModelRegistryPlugin):
    """MLflow model registry adapter.

    Logs training runs with metrics, parameters, models, and artifacts to MLflow.
    Supports loading models and artifacts for serving pipelines.

    Configuration (in profiles.yaml):
        mlflow:
          tracking_uri: "sqlite:///mlruns.db"  # Local SQLite
          # or: "https://mlflow.myorg.com"     # Remote server
          experiment_name: "my_experiment"     # Optional

    Example:
        >>> registry = MLflowRegistry()
        >>> run_id = registry.log_run(
        ...     pipeline_name="churn_v1",
        ...     metrics={"roc_auc": 0.87},
        ...     params={"n_estimators": 100},
        ...     artifacts={"model": model, "scaler": scaler},
        ...     tags={"framework": "sklearn"}
        ... )
    """

    def __init__(self, tracking_uri: str | None = None, experiment_name: str | None = None):
        """Initialize MLflow registry.

        Args:
            tracking_uri: MLflow tracking server URI (default: uses MLFLOW_TRACKING_URI env var)
            experiment_name: MLflow experiment name (default: "mbt")
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name or "mbt"

        # Ensure experiment exists
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)

    def log_run(
        self,
        pipeline_name: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifacts: dict[str, Any],
        tags: dict[str, str],
    ) -> str:
        """Log a training run to MLflow.

        Args:
            pipeline_name: Name of the pipeline
            metrics: Evaluation metrics
            params: Training parameters
            artifacts: Objects to save (model, scaler, encoder, etc.)
            tags: Metadata tags

        Returns:
            MLflow run_id
        """
        # Set experiment
        mlflow.set_experiment(self.experiment_name)

        # Start run
        with mlflow.start_run(run_name=pipeline_name) as run:
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log parameters (convert all to strings for MLflow)
            for key, value in params.items():
                mlflow.log_param(key, str(value))

            # Log tags
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            # Set pipeline_name tag
            mlflow.set_tag("pipeline_name", pipeline_name)

            # Log artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                for artifact_name, artifact_obj in artifacts.items():
                    # Special handling for 'model' - log as MLflow model
                    if artifact_name == "model":
                        try:
                            # Try to log as pyfunc model (works for sklearn, h2o, etc.)
                            mlflow.pyfunc.log_model(
                                artifact_path="model",
                                python_model=None,
                                artifacts=None,
                                code_path=None,
                            )
                        except Exception:
                            # Fallback: save as pickle
                            model_path = temp_path / "model.pkl"
                            with open(model_path, "wb") as f:
                                pickle.dump(artifact_obj, f)
                            mlflow.log_artifact(str(model_path), artifact_path="model")
                    else:
                        # Save other artifacts as pickle
                        artifact_path = temp_path / f"{artifact_name}.pkl"
                        with open(artifact_path, "wb") as f:
                            pickle.dump(artifact_obj, f)
                        mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")

            return run.info.run_id

    def load_model(self, run_id: str) -> Any:
        """Load model from MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Trained model object

        Raises:
            ValueError: If run_id not found or model not available
        """
        try:
            # Try to load as pyfunc model
            model_uri = f"runs:/{run_id}/model"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # Fallback: load from artifacts
            try:
                client = mlflow.tracking.MlflowClient()
                artifact_path = client.download_artifacts(run_id, "model/model.pkl")
                with open(artifact_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load model from run {run_id}: {e}")

    def load_artifacts(self, run_id: str) -> dict[str, Any]:
        """Load all artifacts (except model) from MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of artifact names to objects

        Raises:
            ValueError: If run_id not found
        """
        client = mlflow.tracking.MlflowClient()
        result = {}

        # Try listing artifacts directory first
        try:
            artifacts_list = client.list_artifacts(run_id, path="artifacts")

            for artifact_info in artifacts_list:
                artifact_path = client.download_artifacts(run_id, artifact_info.path)
                with open(artifact_path, "rb") as f:
                    artifact_obj = pickle.load(f)
                artifact_name = Path(artifact_info.path).stem
                result[artifact_name] = artifact_obj

            return result
        except Exception:
            pass

        # Fallback: try downloading known artifact names directly
        # (some MLflow versions/backends don't support list_artifacts reliably)
        known_artifacts = ["feature_selector", "scaler", "encoder"]
        for name in known_artifacts:
            try:
                artifact_path = client.download_artifacts(run_id, f"artifacts/{name}.pkl")
                with open(artifact_path, "rb") as f:
                    result[name] = pickle.load(f)
            except Exception:
                continue

        return result

    def download_artifacts(self, run_id: str, output_dir: str) -> dict[str, str]:
        """Download all artifacts from MLflow run to local directory.

        Args:
            run_id: MLflow run ID
            output_dir: Local directory to download to

        Returns:
            Dictionary mapping artifact names to local file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        client = mlflow.tracking.MlflowClient()

        # Download model
        result = {}
        try:
            model_dir = client.download_artifacts(run_id, "model")
            dest_model_dir = output_path / "model"
            if Path(model_dir).exists():
                if dest_model_dir.exists():
                    shutil.rmtree(dest_model_dir)
                shutil.copytree(model_dir, dest_model_dir)
                result["model"] = str(dest_model_dir)
        except Exception:
            pass

        # Download other artifacts
        try:
            artifacts_list = client.list_artifacts(run_id, path="artifacts")
            for artifact_info in artifacts_list:
                artifact_path = client.download_artifacts(run_id, artifact_info.path)
                artifact_name = Path(artifact_info.path).stem

                # Copy to output directory
                dest_path = output_path / f"{artifact_name}.pkl"
                shutil.copy(artifact_path, dest_path)
                result[artifact_name] = str(dest_path)
        except Exception:
            pass

        return result

    def get_run_info(self, run_id: str) -> dict[str, Any]:
        """Get metadata about MLflow run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary with run information
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }

    def list_runs(
        self, pipeline_name: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """List recent MLflow runs.

        Args:
            pipeline_name: Filter by pipeline name tag
            limit: Maximum number of runs to return

        Returns:
            List of run info dictionaries
        """
        client = mlflow.tracking.MlflowClient()

        # Get experiment ID
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        # Build filter string
        filter_string = ""
        if pipeline_name:
            filter_string = f"tags.pipeline_name = '{pipeline_name}'"

        # Search runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"],
        )

        return [
            {
                "run_id": run.info.run_id,
                "pipeline_name": run.data.tags.get("pipeline_name", ""),
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
            for run in runs
        ]

    def validate_connection(self) -> bool:
        """Validate connection to MLflow tracking server.

        Returns:
            True if connection is successful
        """
        try:
            # Try to list experiments
            mlflow.search_experiments()
            return True
        except Exception:
            return False
