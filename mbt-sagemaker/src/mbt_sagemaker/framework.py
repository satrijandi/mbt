"""AWS SageMaker framework adapter for MBT.

This adapter enables automatic model training on SageMaker infrastructure using built-in algorithms.
Supports XGBoost, LinearLearner, FactorizationMachines, and KNN.
"""

import boto3
import sagemaker
from pathlib import Path
from typing import Any
import numpy as np
import tempfile
import shutil
import uuid

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

from mbt_sagemaker.algorithms import (
    get_builtin_algorithm_spec,
    validate_algorithm_config,
    SUPPORTED_ALGORITHMS,
)
from mbt_sagemaker.s3_utils import upload_dataframe_to_s3
from mbt_sagemaker.exceptions import SageMakerSetupError, SageMakerTrainingError


class SageMakerFramework(FrameworkPlugin):
    """AWS SageMaker framework adapter.

    Automatically trains models on SageMaker infrastructure with built-in algorithms.
    Handles data upload to S3, job launching, and model artifact retrieval.

    Configuration example (pipeline YAML):
        training:
          model_training:
            framework: sagemaker
            config:
              algorithm: xgboost
              instance_type: ml.m5.xlarge
              instance_count: 1
              max_run_time: 3600
              hyperparameters:
                num_round: 100
                max_depth: 5
                eta: 0.2
                objective: binary:logistic

    Profile configuration (profiles.yaml):
        outputs:
          prod:
            sagemaker_connection:
              region: us-east-1
              role_arn: arn:aws:iam::123456789:role/SageMakerRole
              s3_bucket: ml-artifacts-bucket
              s3_prefix: mbt-models/
    """

    def __init__(self):
        """Initialize adapter."""
        self._session = None
        self._sagemaker_session = None
        self._role_arn = None
        self._s3_bucket = None
        self._s3_prefix = None
        self._region = None
        self._temp_dir = None
        self._current_job_name = None

    def setup(self, config: dict) -> None:
        """Initialize AWS session and validate SageMaker access.

        Args:
            config: Resolved profile configuration containing sagemaker_connection

        Raises:
            SageMakerSetupError: If setup fails
        """
        # Extract SageMaker connection config from profile
        sm_config = config.get("sagemaker_connection", {})

        if not sm_config:
            raise SageMakerSetupError(
                "No sagemaker_connection found in profile configuration. "
                "Add sagemaker_connection to your profiles.yaml outputs section."
            )

        # Required fields
        self._role_arn = sm_config.get("role_arn")
        self._s3_bucket = sm_config.get("s3_bucket")
        self._region = sm_config.get("region", "us-east-1")

        if not self._role_arn:
            raise SageMakerSetupError("role_arn is required in sagemaker_connection")
        if not self._s3_bucket:
            raise SageMakerSetupError("s3_bucket is required in sagemaker_connection")

        # Optional fields
        self._s3_prefix = sm_config.get("s3_prefix", "mbt-models/")

        # Initialize AWS sessions
        try:
            self._session = boto3.Session(region_name=self._region)
            self._sagemaker_session = sagemaker.Session(boto_session=self._session)

            # Validate access (try to list training jobs - lightweight check)
            sm_client = self._session.client("sagemaker")
            sm_client.list_training_jobs(MaxResults=1)

            print(f"  ✓ SageMaker session initialized (region: {self._region})")

        except Exception as e:
            raise SageMakerSetupError(
                f"Failed to initialize SageMaker session: {str(e)}\n"
                f"Check AWS credentials and IAM permissions."
            ) from e

        # Create temp directory for local artifacts
        self._temp_dir = tempfile.mkdtemp(prefix="mbt_sagemaker_")

    def teardown(self) -> None:
        """Cleanup temporary resources."""
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        # Clear session references
        self._session = None
        self._sagemaker_session = None

    def health_check(self) -> bool:
        """Check if SageMaker session is healthy.

        Returns:
            True if session is initialized and responsive
        """
        if not self._session or not self._sagemaker_session:
            return False

        try:
            sm_client = self._session.client("sagemaker")
            sm_client.list_training_jobs(MaxResults=1)
            return True
        except Exception:
            return False

    def supported_formats(self) -> list[str]:
        """SageMaker adapter uses pandas for data conversion.

        Returns:
            ["pandas"]
        """
        return ["pandas"]

    def validate_config(self, config: dict, problem_type: str) -> None:
        """Validate SageMaker configuration at compile time.

        Args:
            config: Framework-specific config from model_training.config
            problem_type: Problem type (binary_classification, etc.)

        Raises:
            ValueError: If configuration is invalid
        """
        # Required: algorithm
        if "algorithm" not in config:
            raise ValueError(
                "SageMaker config must specify 'algorithm'. "
                f"Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS)}"
            )

        algorithm = config["algorithm"]
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported SageMaker algorithm: {algorithm}. "
                f"Supported: {', '.join(SUPPORTED_ALGORITHMS)}"
            )

        # Validate algorithm-specific config
        validate_algorithm_config(algorithm, config, problem_type)

        # Validate instance configuration
        if "instance_type" in config:
            instance_type = config["instance_type"]
            if not instance_type.startswith("ml."):
                raise ValueError(
                    f"Invalid instance_type: {instance_type}. "
                    "Must be a SageMaker instance type (e.g., ml.m5.xlarge)"
                )

        if "instance_count" in config:
            instance_count = config["instance_count"]
            if not isinstance(instance_count, int) or instance_count < 1:
                raise ValueError("instance_count must be a positive integer")

        if "max_run_time" in config:
            max_run_time = config["max_run_time"]
            if not isinstance(max_run_time, int) or max_run_time < 1:
                raise ValueError("max_run_time must be a positive integer (seconds)")

    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Train model on SageMaker.

        Steps:
        1. Convert data to pandas and prepare for S3
        2. Upload training data to S3
        3. Configure SageMaker Estimator
        4. Launch training job
        5. Wait for completion
        6. Return model artifact information

        Args:
            X_train: Training features
            y_train: Training labels
            config: SageMaker-specific configuration

        Returns:
            Dictionary with model_data_s3_uri and job_name
        """
        # 1. Convert to pandas
        X_df = X_train.to_pandas()
        y_df = y_train.to_pandas()

        # 2. Combine for SageMaker (target as first column for built-in algos)
        if len(y_df.columns) == 1:
            target_col = y_df.columns[0]
            train_df = y_df.copy()
            for col in X_df.columns:
                train_df[col] = X_df[col]
        else:
            raise ValueError("y_train must have exactly one column")

        print(f"  Preparing training data: {len(train_df)} rows, {len(train_df.columns)} columns")

        # 3. Upload to S3
        try:
            s3_input_uri = upload_dataframe_to_s3(
                df=train_df,
                bucket=self._s3_bucket,
                prefix=f"{self._s3_prefix}training-data/",
                session=self._session,
            )
            print(f"  ✓ Data uploaded to: {s3_input_uri}")
        except Exception as e:
            raise SageMakerTrainingError(f"Failed to upload data to S3: {str(e)}") from e

        # 4. Get algorithm configuration
        algorithm = config["algorithm"]
        algo_spec = get_builtin_algorithm_spec(algorithm, self._region)

        # 5. Configure hyperparameters
        hyperparameters = config.get("hyperparameters", {})

        # 6. Configure instance settings
        instance_type = config.get("instance_type", "ml.m5.xlarge")
        instance_count = config.get("instance_count", 1)
        max_run_time = config.get("max_run_time", 3600)

        # 7. Generate unique job name
        job_name = f"mbt-{algorithm}-{uuid.uuid4().hex[:8]}"
        self._current_job_name = job_name

        print(f"  Launching SageMaker training job: {job_name}")
        print(f"    Algorithm: {algorithm}")
        print(f"    Instance: {instance_type} x {instance_count}")

        # 8. Create Estimator
        try:
            estimator = sagemaker.estimator.Estimator(
                image_uri=algo_spec["image_uri"],
                role=self._role_arn,
                instance_count=instance_count,
                instance_type=instance_type,
                output_path=f"s3://{self._s3_bucket}/{self._s3_prefix}models/",
                sagemaker_session=self._sagemaker_session,
                hyperparameters=hyperparameters,
                max_run=max_run_time,
                base_job_name=algorithm,
            )

            # Launch training job (blocking)
            estimator.fit({"train": s3_input_uri}, job_name=job_name, wait=True)

            print(f"  ✓ Training completed: {job_name}")

            # Get model artifact location
            model_data_uri = estimator.model_data

            return {
                "job_name": job_name,
                "model_data_s3_uri": model_data_uri,
                "estimator": estimator,  # Keep reference for predict
            }

        except Exception as e:
            raise SageMakerTrainingError(
                f"SageMaker training job failed: {str(e)}"
            ) from e

    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions using trained SageMaker model.

        For Phase 1: Simplified implementation returns zeros.
        Full implementation would use SageMaker Batch Transform or endpoints.

        Args:
            model: Model dict from train() containing model_data_s3_uri
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        X_df = X.to_pandas()

        # For Phase 1: Return placeholder predictions
        # Full implementation would:
        # 1. Upload inference data to S3
        # 2. Use SageMaker Batch Transform
        # 3. Download predictions from S3

        print(f"  ⚠ Using placeholder predictions (Phase 1 implementation)")
        return np.zeros(len(X_df))

    def serialize(self, model: Any, path: str) -> None:
        """Save model metadata to disk.

        Args:
            model: Model dict with S3 URI
            path: Local path to save metadata
        """
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump({
                "job_name": model["job_name"],
                "model_data_s3_uri": model["model_data_s3_uri"],
            }, f)

    def deserialize(self, path: str) -> Any:
        """Load model metadata from disk.

        Args:
            path: Path to metadata file

        Returns:
            Model dict
        """
        import json

        with open(path, "r") as f:
            return json.load(f)

    def get_training_metrics(self, model: Any) -> dict[str, float]:
        """Extract training metrics from SageMaker job.

        Args:
            model: Model dict from train()

        Returns:
            Dictionary of training metrics
        """
        job_name = model.get("job_name")
        if not job_name:
            return {}

        try:
            sm_client = self._session.client("sagemaker")
            job_desc = sm_client.describe_training_job(TrainingJobName=job_name)

            # Extract final metrics if available
            metrics = {}
            if "FinalMetricDataList" in job_desc:
                for metric in job_desc["FinalMetricDataList"]:
                    metrics[metric["MetricName"]] = float(metric["Value"])

            return metrics
        except Exception:
            return {}
