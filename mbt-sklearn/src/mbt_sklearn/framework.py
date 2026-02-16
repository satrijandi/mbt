"""scikit-learn framework adapter for MBT."""

import joblib
from typing import Any
import numpy as np
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# Import FrameworkPlugin from mbt-core
try:
    from mbt.contracts.framework import FrameworkPlugin
    from mbt.core.data import MBTFrame
except ImportError:
    # Fallback for development/testing
    class FrameworkPlugin:
        """Stub for development."""
        pass

    class MBTFrame:
        """Stub for development."""
        pass


# Supported models by problem type
CLASSIFICATION_MODELS = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "SVC": SVC,
}

REGRESSION_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
}


class SklearnFramework(FrameworkPlugin):
    """scikit-learn framework adapter.

    Supports common sklearn models for classification and regression.

    Configuration example (binary classification):
        {
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }

    Configuration example (regression):
        {
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 15
        }
    """

    def validate_config(self, config: dict, problem_type: str) -> None:
        """Validate sklearn configuration at compile time.

        Args:
            config: Framework configuration
            problem_type: Problem type (binary_classification, multiclass_classification, regression)

        Raises:
            ValueError: If configuration is invalid
        """
        if "model" not in config:
            raise ValueError("sklearn config must specify 'model' parameter")

        model_name = config["model"]

        # Check if model is supported for this problem type
        if problem_type in ["binary_classification", "multiclass_classification"]:
            if model_name not in CLASSIFICATION_MODELS:
                supported = ", ".join(CLASSIFICATION_MODELS.keys())
                raise ValueError(
                    f"Unsupported sklearn classification model: {model_name}. "
                    f"Supported models: {supported}"
                )
        elif problem_type == "regression":
            if model_name not in REGRESSION_MODELS:
                supported = ", ".join(REGRESSION_MODELS.keys())
                raise ValueError(
                    f"Unsupported sklearn regression model: {model_name}. "
                    f"Supported models: {supported}"
                )
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        # Validate model-specific parameters
        model_class = self._get_model_class(model_name, problem_type)
        valid_params = set(model_class().get_params().keys())

        for param in config:
            if param == "model":
                continue
            if param not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{param}' for {model_name}. "
                    f"Valid parameters: {sorted(valid_params)}"
                )

    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Train sklearn model.

        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration

        Returns:
            Trained sklearn model
        """
        # Convert to pandas
        X_df = X_train.to_pandas()
        y_df = y_train.to_pandas()

        # Extract target column (assuming single column)
        if len(y_df.columns) == 1:
            y = y_df.iloc[:, 0]
        else:
            y = y_df

        # Determine problem type from y
        n_unique = len(y.unique())
        if n_unique == 2:
            problem_type = "binary_classification"
        elif n_unique > 2 and y.dtype in ['int64', 'object', 'category']:
            problem_type = "multiclass_classification"
        else:
            problem_type = "regression"

        # Get model class
        model_name = config["model"]
        model_class = self._get_model_class(model_name, problem_type)

        # Extract model parameters (exclude 'model' key)
        model_params = {k: v for k, v in config.items() if k != "model"}

        # Set random_state if not provided (for reproducibility)
        if "random_state" not in model_params and hasattr(model_class(), "random_state"):
            model_params["random_state"] = 42

        # Instantiate and train
        model = model_class(**model_params)
        model.fit(X_df, y)

        return model

    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            model: Trained sklearn model
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        X_df = X.to_pandas()
        return model.predict(X_df)

    def predict_proba(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate probability predictions (for classification).

        Args:
            model: Trained sklearn model
            X: Features to predict on

        Returns:
            Probability predictions

        Raises:
            AttributeError: If model doesn't support predict_proba
        """
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"{model.__class__.__name__} does not support probability predictions"
            )

        X_df = X.to_pandas()
        return model.predict_proba(X_df)

    def serialize(self, model: Any, path: str) -> None:
        """Save model to disk using joblib.

        Args:
            model: Trained sklearn model
            path: File path to save to
        """
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    def deserialize(self, path: str) -> Any:
        """Load model from disk.

        Args:
            path: File path to load from

        Returns:
            Trained sklearn model
        """
        return joblib.load(path)

    def get_feature_importance(self, model: Any) -> dict[str, float] | None:
        """Extract feature importance from tree-based models.

        Args:
            model: Trained sklearn model

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if not hasattr(model, "feature_importances_"):
            return None

        # Note: feature names would need to be passed separately
        # For now, return indices
        importances = model.feature_importances_
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

    def get_training_metrics(self, model: Any) -> dict[str, float]:
        """Extract training metrics from model.

        Args:
            model: Trained sklearn model

        Returns:
            Dictionary of training metrics
        """
        # sklearn models don't store training metrics by default
        # Could be extended to store score during training
        return {}

    def _get_model_class(self, model_name: str, problem_type: str):
        """Get sklearn model class by name and problem type.

        Args:
            model_name: Model class name
            problem_type: Problem type

        Returns:
            sklearn model class

        Raises:
            ValueError: If model not found
        """
        if problem_type in ["binary_classification", "multiclass_classification"]:
            if model_name in CLASSIFICATION_MODELS:
                return CLASSIFICATION_MODELS[model_name]
        elif problem_type == "regression":
            if model_name in REGRESSION_MODELS:
                return REGRESSION_MODELS[model_name]

        raise ValueError(f"Model {model_name} not supported for {problem_type}")
