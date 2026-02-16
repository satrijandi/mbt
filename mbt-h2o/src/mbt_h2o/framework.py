"""H2O AutoML framework adapter for MBT.

This adapter enables automatic model training and hyperparameter tuning using H2O AutoML.
H2O AutoML trains and tunes multiple models (GLM, GBM, DRF, XGBoost, DeepLearning, StackedEnsemble)
and automatically selects the best performer.
"""

import h2o
from h2o.automl import H2OAutoML
from typing import Any
import numpy as np
from pathlib import Path
import tempfile
import shutil

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


# Valid sort metrics by problem type
CLASSIFICATION_METRICS = ["AUC", "AUCPR", "logloss", "mean_per_class_error"]
REGRESSION_METRICS = ["deviance", "RMSE", "MSE", "MAE", "RMSLE"]


class H2OAutoMLFramework(FrameworkPlugin):
    """H2O AutoML framework adapter.

    Automatically trains and tunes multiple models and selects the best one.
    This is the AutoML-first approach: strong models without manual hyperparameter tuning.

    Configuration example:
        {
            "max_runtime_secs": 3600,    # 1 hour training budget
            "max_models": 20,            # Train up to 20 models
            "sort_metric": "AUC",        # Optimize for ROC AUC
            "seed": 42,                  # For reproducibility
            "nfolds": 5,                 # Cross-validation folds
            "balance_classes": true      # For imbalanced classification
        }

    H2O AutoML will train:
    - GLM (Generalized Linear Model)
    - Random Forest (DRF)
    - Gradient Boosting Machine (GBM)
    - XGBoost
    - Deep Learning
    - Stacked Ensemble (combines best models)
    """

    def __init__(self):
        self._h2o_initialized = False
        self._temp_dir = None

    def setup(self, config: dict) -> None:
        """Initialize H2O cluster.

        Args:
            config: Full pipeline configuration
        """
        if not self._h2o_initialized:
            # Initialize H2O with reasonable defaults
            h2o.init(
                max_mem_size="4G",  # Can be overridden via H2O_MAX_MEM_SIZE env var
                strict_version_check=False,
            )
            self._h2o_initialized = True

    def teardown(self) -> None:
        """Shutdown H2O cluster and cleanup temporary files."""
        if self._h2o_initialized:
            h2o.cluster().shutdown(prompt=False)
            self._h2o_initialized = False

        # Cleanup temporary directory
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def health_check(self) -> bool:
        """Check if H2O cluster is running.

        Returns:
            True if H2O is initialized and healthy
        """
        try:
            if self._h2o_initialized:
                cluster = h2o.cluster()
                return cluster is not None and cluster.is_running()
            return False
        except Exception:
            return False

    def supported_formats(self) -> list[str]:
        """H2O supports both pandas and native H2O frames.

        Returns:
            ["pandas", "h2o"]
        """
        return ["pandas", "h2o"]

    def validate_config(self, config: dict, problem_type: str) -> None:
        """Validate H2O AutoML configuration at compile time.

        Args:
            config: Framework configuration
            problem_type: Problem type

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate sort_metric for problem type
        if "sort_metric" in config:
            metric = config["sort_metric"]

            if problem_type in ["binary_classification", "multiclass_classification"]:
                if metric not in CLASSIFICATION_METRICS:
                    supported = ", ".join(CLASSIFICATION_METRICS)
                    raise ValueError(
                        f"Invalid sort_metric '{metric}' for {problem_type}. "
                        f"Supported metrics: {supported}"
                    )
            elif problem_type == "regression":
                if metric not in REGRESSION_METRICS:
                    supported = ", ".join(REGRESSION_METRICS)
                    raise ValueError(
                        f"Invalid sort_metric '{metric}' for {problem_type}. "
                        f"Supported metrics: {supported}"
                    )

        # Validate numeric parameters
        if "max_runtime_secs" in config:
            if not isinstance(config["max_runtime_secs"], (int, float)) or config["max_runtime_secs"] <= 0:
                raise ValueError("max_runtime_secs must be a positive number")

        if "max_models" in config:
            if not isinstance(config["max_models"], int) or config["max_models"] <= 0:
                raise ValueError("max_models must be a positive integer")

        if "nfolds" in config:
            if not isinstance(config["nfolds"], int) or config["nfolds"] < 0:
                raise ValueError("nfolds must be a non-negative integer")

    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Train models using H2O AutoML.

        Args:
            X_train: Training features
            y_train: Training labels
            config: H2O AutoML configuration

        Returns:
            Best model from AutoML (H2OAutoML.leader)
        """
        # Convert to pandas
        X_df = X_train.to_pandas()
        y_df = y_train.to_pandas()

        # Get target column name (assuming single column)
        if len(y_df.columns) == 1:
            target_col = y_df.columns[0]
        else:
            raise ValueError("y_train must have exactly one column")

        # Combine X and y for H2O
        train_df = X_df.copy()
        train_df[target_col] = y_df[target_col]

        # Convert to H2OFrame
        h2o_train = h2o.H2OFrame(train_df)

        # Set target as factor for classification
        if target_col in h2o_train.columns:
            n_unique = h2o_train[target_col].nunique()
            if n_unique <= 20:  # Heuristic: <= 20 unique values = classification
                h2o_train[target_col] = h2o_train[target_col].asfactor()

        # Extract AutoML parameters
        max_runtime_secs = config.get("max_runtime_secs", 3600)  # Default: 1 hour
        max_models = config.get("max_models", 20)
        sort_metric = config.get("sort_metric", "AUTO")
        seed = config.get("seed", 42)
        nfolds = config.get("nfolds", 5)
        balance_classes = config.get("balance_classes", False)

        # Feature columns (all except target)
        feature_cols = [col for col in h2o_train.columns if col != target_col]

        # Train AutoML
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            sort_metric=sort_metric,
            seed=seed,
            nfolds=nfolds,
            balance_classes=balance_classes,
            verbosity="info",
        )

        aml.train(
            x=feature_cols,
            y=target_col,
            training_frame=h2o_train,
        )

        # Return the best model
        return aml.leader

    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions using H2O model.

        Args:
            model: Trained H2O model
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        X_df = X.to_pandas()
        h2o_test = h2o.H2OFrame(X_df)

        # Generate predictions
        preds = model.predict(h2o_test)

        # Convert to numpy
        # For classification, H2O returns [predict, p0, p1, ...] columns
        # We want just the 'predict' column
        pred_col = preds[:, 0]  # First column is the prediction
        return pred_col.as_data_frame().values.flatten()

    def predict_proba(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate probability predictions for classification.

        Args:
            model: Trained H2O model
            X: Features to predict on

        Returns:
            Probability predictions as numpy array
        """
        X_df = X.to_pandas()
        h2o_test = h2o.H2OFrame(X_df)

        # Generate predictions
        preds = model.predict(h2o_test)

        # For binary classification: columns are [predict, p0, p1]
        # For multiclass: columns are [predict, p0, p1, p2, ...]
        # We want the probability columns (skip first column)
        if preds.ncol > 1:
            prob_cols = preds[:, 1:]  # All columns except first
            return prob_cols.as_data_frame().values
        else:
            raise ValueError("Model does not support probability predictions")

    def serialize(self, model: Any, path: str) -> None:
        """Save H2O model to disk.

        Args:
            model: Trained H2O model
            path: Directory path to save model to (H2O saves as directory)
        """
        # H2O saves models as directories, not single files
        parent_dir = Path(path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = h2o.save_model(model=model, path=str(parent_dir), force=True)

        # H2O creates a directory with the model name
        # Move it to the desired path if different
        if model_path != path:
            if Path(path).exists():
                shutil.rmtree(path)
            shutil.move(model_path, path)

    def deserialize(self, path: str) -> Any:
        """Load H2O model from disk.

        Args:
            path: Path to saved H2O model directory

        Returns:
            Trained H2O model
        """
        return h2o.load_model(path)

    def get_feature_importance(self, model: Any) -> dict[str, float] | None:
        """Extract feature importance from H2O model.

        Args:
            model: Trained H2O model

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        try:
            # Get variable importance
            varimp = model.varimp(use_pandas=True)
            if varimp is not None and len(varimp) > 0:
                # varimp is a DataFrame with columns: variable, relative_importance, scaled_importance, percentage
                return dict(zip(varimp["variable"], varimp["scaled_importance"]))
            return None
        except Exception:
            return None

    def get_training_metrics(self, model: Any) -> dict[str, float]:
        """Extract training metrics from H2O model.

        Args:
            model: Trained H2O model

        Returns:
            Dictionary of training metrics
        """
        try:
            # Get model metrics
            metrics = {}

            # Try to get AUC for classification
            if hasattr(model, "auc"):
                try:
                    metrics["train_auc"] = float(model.auc(train=True))
                except Exception:
                    pass

            # Try to get logloss
            if hasattr(model, "logloss"):
                try:
                    metrics["train_logloss"] = float(model.logloss(train=True))
                except Exception:
                    pass

            # Try to get MSE for regression
            if hasattr(model, "mse"):
                try:
                    metrics["train_mse"] = float(model.mse(train=True))
                except Exception:
                    pass

            # Try to get RMSE
            if hasattr(model, "rmse"):
                try:
                    metrics["train_rmse"] = float(model.rmse(train=True))
                except Exception:
                    pass

            return metrics
        except Exception:
            return {}
