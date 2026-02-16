"""Model registry plugin contract for experiment tracking and model management.

Model registry plugins allow MBT to integrate with experiment tracking systems
(MLflow, Weights & Biases, Neptune, etc.) for logging runs, models, and artifacts.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelRegistryPlugin(ABC):
    """Abstract base class for model registry adapters.

    Model registry plugins handle:
    - Logging training runs with metrics, parameters, and tags
    - Saving models and artifacts
    - Loading models and artifacts for serving
    - Versioning and lineage tracking

    Example:
        >>> registry = plugin_registry.get("mbt.model_registries", "mlflow")
        >>> run_id = registry.log_run(
        ...     pipeline_name="churn_v1",
        ...     metrics={"roc_auc": 0.87, "accuracy": 0.82},
        ...     params={"n_estimators": 100, "max_depth": 10},
        ...     artifacts={"model": model_obj, "scaler": scaler_obj},
        ...     tags={"framework": "sklearn", "problem_type": "binary_classification"}
        ... )
        >>> print(f"Logged run: {run_id}")
        >>>
        >>> # Later, in serving pipeline
        >>> model = registry.load_model(run_id)
        >>> artifacts = registry.load_artifacts(run_id)
    """

    @abstractmethod
    def log_run(
        self,
        pipeline_name: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifacts: dict[str, Any],
        tags: dict[str, str],
    ) -> str:
        """Log a training run to the model registry.

        This is called at the end of training to record all metrics, parameters,
        model, and artifacts for future reference and serving.

        Args:
            pipeline_name: Name of the pipeline that produced this run
            metrics: Evaluation metrics (e.g., {"roc_auc": 0.87, "accuracy": 0.82})
            params: Training parameters (e.g., {"n_estimators": 100, "max_depth": 10})
            artifacts: Objects to save (e.g., {"model": model, "scaler": scaler, "encoder": encoder})
            tags: Metadata tags (e.g., {"framework": "sklearn", "problem_type": "binary_classification"})

        Returns:
            run_id: Unique identifier for this run (used for loading in serving)

        Example:
            >>> run_id = registry.log_run(
            ...     pipeline_name="churn_training_v1",
            ...     metrics={"roc_auc": 0.87, "f1": 0.75, "precision": 0.80},
            ...     params={"framework": "sklearn", "model": "RandomForest", "n_estimators": 100},
            ...     artifacts={
            ...         "model": trained_model,
            ...         "scaler": standard_scaler,
            ...         "feature_selector": selector,
            ...     },
            ...     tags={
            ...         "framework": "sklearn",
            ...         "problem_type": "binary_classification",
            ...         "target": "prod",
            ...     }
            ... )
            >>> run_id
            'abc123def456'
        """
        pass

    @abstractmethod
    def load_model(self, run_id: str) -> Any:
        """Load the trained model from a specific run.

        Used in serving pipelines to load the model trained in a previous run.

        Args:
            run_id: Unique run identifier (returned from log_run())

        Returns:
            Trained model object (framework-specific type)

        Raises:
            ValueError: If run_id not found

        Example:
            >>> model = registry.load_model("abc123def456")
            >>> predictions = model.predict(X_test)
        """
        pass

    @abstractmethod
    def load_artifacts(self, run_id: str) -> dict[str, Any]:
        """Load all artifacts (except model) from a specific run.

        Artifacts include preprocessing objects (scalers, encoders, feature selectors)
        that were saved during training and are needed for serving.

        Args:
            run_id: Unique run identifier

        Returns:
            Dictionary mapping artifact names to loaded objects

        Raises:
            ValueError: If run_id not found

        Example:
            >>> artifacts = registry.load_artifacts("abc123def456")
            >>> artifacts.keys()
            dict_keys(['scaler', 'encoder', 'feature_selector'])
            >>> scaler = artifacts['scaler']
            >>> X_scaled = scaler.transform(X_test)
        """
        pass

    def download_artifacts(self, run_id: str, output_dir: str) -> dict[str, str]:
        """Download all artifacts from a run to a local directory.

        This is used during compilation of serving pipelines to create an
        artifact snapshot that can be deployed without runtime dependencies
        on the model registry.

        Args:
            run_id: Unique run identifier
            output_dir: Local directory to download artifacts to

        Returns:
            Dictionary mapping artifact names to local file paths

        Note:
            Default implementation raises NotImplementedError. Override to
            support artifact snapshots in serving pipelines.

        Example:
            >>> paths = registry.download_artifacts(
            ...     run_id="abc123def456",
            ...     output_dir="./target/serving_v1/artifacts/"
            ... )
            >>> paths
            {
                'model': './target/serving_v1/artifacts/model.pkl',
                'scaler': './target/serving_v1/artifacts/scaler.pkl',
                'encoder': './target/serving_v1/artifacts/encoder.pkl',
            }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support artifact downloads"
        )

    def get_run_info(self, run_id: str) -> dict[str, Any]:
        """Get metadata about a specific run.

        Args:
            run_id: Unique run identifier

        Returns:
            Dictionary with run information (metrics, params, tags, timestamps, etc.)

        Note:
            Default implementation returns empty dict. Override to provide run metadata.

        Example:
            >>> info = registry.get_run_info("abc123def456")
            >>> info
            {
                'run_id': 'abc123def456',
                'pipeline_name': 'churn_training_v1',
                'start_time': '2026-02-15T10:30:00',
                'end_time': '2026-02-15T10:35:00',
                'status': 'FINISHED',
                'metrics': {'roc_auc': 0.87, 'accuracy': 0.82},
                'params': {'n_estimators': 100, 'max_depth': 10},
                'tags': {'framework': 'sklearn'},
            }
        """
        return {}

    def list_runs(
        self, pipeline_name: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """List recent runs, optionally filtered by pipeline name.

        Args:
            pipeline_name: Filter by pipeline name (None = all pipelines)
            limit: Maximum number of runs to return

        Returns:
            List of run info dictionaries, sorted by start time (newest first)

        Note:
            Default implementation returns empty list. Override to support run listing.

        Example:
            >>> runs = registry.list_runs(pipeline_name="churn_training_v1", limit=5)
            >>> for run in runs:
            ...     print(f"{run['run_id']}: ROC AUC = {run['metrics']['roc_auc']}")
            abc123: ROC AUC = 0.87
            def456: ROC AUC = 0.85
            ghi789: ROC AUC = 0.83
        """
        return []

    def validate_connection(self) -> bool:
        """Validate connection to the model registry.

        Returns:
            True if registry is accessible, False otherwise

        Note:
            Default implementation always returns True. Override for registries
            that require connectivity checks.
        """
        return True
