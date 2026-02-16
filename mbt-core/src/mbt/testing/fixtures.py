"""Test fixtures for MBT framework testing.

Provides mock objects for testing without dependencies:
- MockMBTFrame: In-memory data frame
- MockStoragePlugin: In-memory storage
- MockFrameworkPlugin: Dummy ML framework
- MockModelRegistry: In-memory model registry
"""

import pickle
from typing import Any
import pandas as pd
import numpy as np

from mbt.core.data import MBTFrame


class MockMBTFrame(MBTFrame):
    """Mock MBTFrame for testing.

    Wraps a pandas DataFrame for testing without real data sources.
    """

    def __init__(self, data: dict | pd.DataFrame):
        """Initialize with dict or DataFrame.

        Args:
            data: Dictionary of column data or pandas DataFrame
        """
        if isinstance(data, dict):
            self._df = pd.DataFrame(data)
        else:
            self._df = data

    def to_pandas(self) -> pd.DataFrame:
        """Return underlying pandas DataFrame."""
        return self._df

    def num_rows(self) -> int:
        """Return number of rows."""
        return len(self._df)

    def columns(self) -> list[str]:
        """Return column names."""
        return list(self._df.columns)

    def schema(self) -> dict[str, str]:
        """Return column types."""
        return {col: str(dtype) for col, dtype in self._df.dtypes.items()}


class MockStoragePlugin:
    """Mock storage plugin for testing.

    Stores artifacts in memory instead of filesystem.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._store: dict[str, bytes] = {}

    def put(
        self,
        artifact_name: str,
        data: bytes,
        run_id: str,
        step_name: str,
        metadata: dict | None = None,
    ) -> str:
        """Store artifact in memory.

        Args:
            artifact_name: Name of artifact
            data: Serialized artifact data
            run_id: Pipeline run ID
            step_name: Step that produced artifact
            metadata: Optional metadata

        Returns:
            URI of stored artifact
        """
        uri = f"mock://{run_id}/{step_name}/{artifact_name}"
        self._store[uri] = data
        return uri

    def get(self, artifact_uri: str) -> bytes:
        """Retrieve artifact from memory.

        Args:
            artifact_uri: URI returned by put()

        Returns:
            Serialized artifact data

        Raises:
            KeyError: If artifact not found
        """
        if artifact_uri not in self._store:
            raise KeyError(f"Artifact not found: {artifact_uri}")
        return self._store[artifact_uri]

    def exists(self, artifact_uri: str) -> bool:
        """Check if artifact exists.

        Args:
            artifact_uri: URI to check

        Returns:
            True if artifact exists
        """
        return artifact_uri in self._store

    def list_artifacts(self, run_id: str) -> list[str]:
        """List all artifacts for a run.

        Args:
            run_id: Pipeline run ID

        Returns:
            List of artifact URIs
        """
        prefix = f"mock://{run_id}/"
        return [uri for uri in self._store.keys() if uri.startswith(prefix)]

    def clear(self):
        """Clear all stored artifacts."""
        self._store.clear()


class MockFrameworkPlugin:
    """Mock framework plugin for testing.

    Implements a simple dummy model for testing pipelines.
    """

    def setup(self, config: dict) -> None:
        """No-op setup."""
        pass

    def teardown(self) -> None:
        """No-op teardown."""
        pass

    def health_check(self) -> bool:
        """Always healthy."""
        return True

    def supported_formats(self) -> list[str]:
        """Support pandas."""
        return ["pandas"]

    def validate_config(self, config: dict, problem_type: str) -> None:
        """No validation - accepts any config."""
        pass

    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Return a dummy model.

        Args:
            X_train: Training features
            y_train: Training labels
            config: Training config

        Returns:
            Dummy model object
        """
        # Create a simple dummy model
        class DummyModel:
            def __init__(self, n_features: int, problem_type: str):
                self.n_features = n_features
                self.problem_type = problem_type

            def predict(self, X: pd.DataFrame) -> np.ndarray:
                # Always predict class 0
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                # 80% probability for class 0
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 0.8
                probs[:, 1] = 0.2
                return probs

        X_df = X_train.to_pandas()
        problem_type = config.get("problem_type", "binary_classification")

        return DummyModel(n_features=len(X_df.columns), problem_type=problem_type)

    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            model: Trained model
            X: Features to predict on

        Returns:
            Predictions array
        """
        X_df = X.to_pandas()
        return model.predict(X_df)

    def predict_proba(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate prediction probabilities.

        Args:
            model: Trained model
            X: Features to predict on

        Returns:
            Probabilities array
        """
        X_df = X.to_pandas()
        return model.predict_proba(X_df)

    def serialize(self, model: Any, path: str) -> None:
        """Serialize model to file.

        Args:
            model: Model to serialize
            path: File path
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def deserialize(self, path: str) -> Any:
        """Deserialize model from file.

        Args:
            path: File path

        Returns:
            Loaded model
        """
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_training_metrics(self, model: Any) -> dict:
        """Return dummy training metrics.

        Args:
            model: Trained model

        Returns:
            Metrics dictionary
        """
        return {
            "train_accuracy": 0.95,
            "train_loss": 0.15,
        }


class MockModelRegistry:
    """Mock model registry for testing.

    Stores models and artifacts in memory.
    """

    def __init__(self):
        """Initialize in-memory registry."""
        self._runs: dict[str, dict] = {}

    def log_run(
        self,
        pipeline_name: str,
        metrics: dict,
        params: dict,
        artifacts: dict,
        tags: dict,
    ) -> str:
        """Log a run to the registry.

        Args:
            pipeline_name: Name of pipeline
            metrics: Metrics dictionary
            params: Parameters dictionary
            artifacts: Artifacts dictionary
            tags: Tags dictionary

        Returns:
            Run ID
        """
        import uuid
        run_id = str(uuid.uuid4())

        self._runs[run_id] = {
            "pipeline_name": pipeline_name,
            "metrics": metrics,
            "params": params,
            "artifacts": artifacts,
            "tags": tags,
        }

        return run_id

    def load_model(self, run_id: str) -> Any:
        """Load model from run.

        Args:
            run_id: Run ID

        Returns:
            Model object

        Raises:
            KeyError: If run not found
        """
        if run_id not in self._runs:
            raise KeyError(f"Run not found: {run_id}")

        artifacts = self._runs[run_id]["artifacts"]
        if "model" not in artifacts:
            raise KeyError(f"No model in run: {run_id}")

        return artifacts["model"]

    def load_artifacts(self, run_id: str) -> dict:
        """Load artifacts from run.

        Args:
            run_id: Run ID

        Returns:
            Artifacts dictionary

        Raises:
            KeyError: If run not found
        """
        if run_id not in self._runs:
            raise KeyError(f"Run not found: {run_id}")

        # Return all artifacts except model
        artifacts = self._runs[run_id]["artifacts"].copy()
        artifacts.pop("model", None)
        return artifacts

    def get_run_info(self, run_id: str) -> dict:
        """Get run metadata.

        Args:
            run_id: Run ID

        Returns:
            Run metadata dictionary

        Raises:
            KeyError: If run not found
        """
        if run_id not in self._runs:
            raise KeyError(f"Run not found: {run_id}")

        run = self._runs[run_id]
        return {
            "run_id": run_id,
            "pipeline_name": run["pipeline_name"],
            "metrics": run["metrics"],
            "params": run["params"],
            "tags": run["tags"],
        }

    def clear(self):
        """Clear all runs."""
        self._runs.clear()
