"""Framework plugin contract for ML training frameworks.

Framework plugins allow MBT to support multiple ML frameworks (sklearn, H2O, XGBoost, etc.)
without hardcoding dependencies in the core package.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from mbt.core.data import MBTFrame


class FrameworkPlugin(ABC):
    """Abstract base class for ML framework adapters.

    Framework plugins encapsulate the logic for training, predicting, and serializing
    models using a specific ML framework (sklearn, H2O AutoML, XGBoost, etc.).

    Lifecycle:
        1. __init__() - Plugin instantiated by registry
        2. validate_config() - Called at compile time to validate configuration
        3. setup() - Called before training to initialize framework
        4. train() - Train the model
        5. predict() - Generate predictions
        6. serialize() / deserialize() - Save/load models
        7. teardown() - Called after training to cleanup resources

    Example:
        >>> framework = registry.get("mbt.frameworks", "sklearn")
        >>> framework.validate_config(config, "binary_classification")
        >>> framework.setup(config)
        >>> model = framework.train(X_train, y_train, config)
        >>> predictions = framework.predict(model, X_test)
        >>> framework.teardown()
    """

    def setup(self, config: dict) -> None:
        """Initialize framework resources before training.

        Called once before training starts. Use this to:
        - Start framework servers (e.g., h2o.init())
        - Configure framework settings
        - Allocate resources

        Args:
            config: Full pipeline configuration

        Note:
            Default implementation is a no-op. Override if framework needs initialization.
        """
        pass

    def teardown(self) -> None:
        """Cleanup framework resources after training.

        Called after training completes (success or failure). Use this to:
        - Shutdown framework servers (e.g., h2o.cluster().shutdown())
        - Release resources
        - Cleanup temporary files

        Note:
            Default implementation is a no-op. Override if framework needs cleanup.
        """
        pass

    def health_check(self) -> bool:
        """Check if framework is available and healthy.

        Returns:
            True if framework is ready to use, False otherwise

        Note:
            Default implementation always returns True. Override for frameworks
            that require connectivity checks (e.g., remote servers).
        """
        return True

    def supported_formats(self) -> list[str]:
        """List of data formats this framework can consume.

        Returns:
            List of format names (e.g., ["pandas", "h2o", "spark"])

        Note:
            Default is ["pandas"]. Override if framework supports other formats.
            This enables MBTFrame format negotiation.
        """
        return ["pandas"]

    @abstractmethod
    def validate_config(self, config: dict, problem_type: str) -> None:
        """Validate framework-specific configuration at compile time.

        This is called during compilation to catch configuration errors early,
        before any training happens.

        Args:
            config: Framework-specific config from model_training.config section
            problem_type: Problem type (binary_classification, multiclass_classification, regression)

        Raises:
            ValueError: If configuration is invalid

        Example:
            For sklearn:
            >>> config = {"model": "InvalidModel", "n_estimators": 100}
            >>> framework.validate_config(config, "binary_classification")
            ValueError: Unsupported sklearn model: InvalidModel

            For H2O:
            >>> config = {"sort_metric": "INVALID"}
            >>> framework.validate_config(config, "binary_classification")
            ValueError: Invalid sort_metric 'INVALID' for binary_classification
        """
        pass

    @abstractmethod
    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Train a model on the provided data.

        Args:
            X_train: Training features as MBTFrame
            y_train: Training labels as MBTFrame
            config: Framework-specific configuration

        Returns:
            Trained model object (framework-specific type)

        Example:
            >>> X_train = PandasFrame(train_df[features])
            >>> y_train = PandasFrame(train_df[[target]])
            >>> config = {"model": "RandomForestClassifier", "n_estimators": 100}
            >>> model = framework.train(X_train, y_train, config)
        """
        pass

    @abstractmethod
    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions using a trained model.

        Args:
            model: Trained model object (returned from train())
            X: Features to predict on as MBTFrame

        Returns:
            Predictions as numpy array

        Example:
            >>> X_test = PandasFrame(test_df[features])
            >>> predictions = framework.predict(model, X_test)
            >>> predictions.shape
            (100,)  # One prediction per row
        """
        pass

    def predict_proba(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate probability predictions (for classification).

        Args:
            model: Trained model object
            X: Features as MBTFrame

        Returns:
            Probability predictions as numpy array
            - Binary: shape (n_samples, 2)
            - Multiclass: shape (n_samples, n_classes)

        Note:
            Default implementation raises NotImplementedError. Override for
            frameworks that support probability predictions.

        Raises:
            NotImplementedError: If framework doesn't support predict_proba
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability predictions"
        )

    @abstractmethod
    def serialize(self, model: Any, path: str) -> None:
        """Save a trained model to disk.

        Args:
            model: Trained model object
            path: File path to save model to

        Example:
            >>> framework.serialize(model, "/tmp/model.pkl")
        """
        pass

    @abstractmethod
    def deserialize(self, path: str) -> Any:
        """Load a trained model from disk.

        Args:
            path: File path to load model from

        Returns:
            Trained model object

        Example:
            >>> model = framework.deserialize("/tmp/model.pkl")
        """
        pass

    def get_feature_importance(self, model: Any) -> dict[str, float] | None:
        """Extract feature importance from trained model.

        Args:
            model: Trained model object

        Returns:
            Dictionary mapping feature names to importance scores, or None if not supported

        Note:
            Default implementation returns None. Override for frameworks that
            support feature importance.
        """
        return None

    def get_training_metrics(self, model: Any) -> dict[str, float]:
        """Extract training metrics from model.

        Args:
            model: Trained model object

        Returns:
            Dictionary of training metrics (e.g., {"train_accuracy": 0.95})

        Note:
            Default implementation returns empty dict. Override to extract
            framework-specific training metrics.
        """
        return {}
