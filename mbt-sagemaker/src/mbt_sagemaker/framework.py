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
    MBT_TO_AUTOPILOT_PROBLEM_TYPE,
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

        Supports two modes:
        - Built-in algorithms (xgboost, linear-learner, etc.): Uses SageMaker Estimator
        - Autopilot: Uses SageMaker AutoML to automatically find the best model

        Args:
            X_train: Training features
            y_train: Training labels
            config: SageMaker-specific configuration

        Returns:
            Dictionary with model info (model_data_s3_uri, job_name, etc.)
        """
        algorithm = config["algorithm"]

        if algorithm == "autopilot":
            return self._train_autopilot(X_train, y_train, config)
        else:
            return self._train_builtin(X_train, y_train, config)

    def _train_builtin(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> dict:
        """Train using a SageMaker built-in algorithm.

        Args:
            X_train: Training features
            y_train: Training labels
            config: Algorithm configuration

        Returns:
            Dictionary with model_data_s3_uri, job_name, and estimator
        """
        X_df = X_train.to_pandas()
        y_df = y_train.to_pandas()

        # Combine for SageMaker (target as first column for built-in algos)
        if len(y_df.columns) == 1:
            target_col = y_df.columns[0]
            train_df = y_df.copy()
            for col in X_df.columns:
                train_df[col] = X_df[col]
        else:
            raise ValueError("y_train must have exactly one column")

        print(f"  Preparing training data: {len(train_df)} rows, {len(train_df.columns)} columns")

        # Upload to S3 (no header for built-in algos)
        try:
            s3_input_uri = upload_dataframe_to_s3(
                df=train_df,
                bucket=self._s3_bucket,
                prefix=f"{self._s3_prefix}training-data/",
                session=self._session,
                include_header=False,
            )
            print(f"  ✓ Data uploaded to: {s3_input_uri}")
        except Exception as e:
            raise SageMakerTrainingError(f"Failed to upload data to S3: {str(e)}") from e

        # Get algorithm configuration
        algorithm = config["algorithm"]
        algo_spec = get_builtin_algorithm_spec(algorithm, self._region)

        hyperparameters = config.get("hyperparameters", {})
        instance_type = config.get("instance_type", "ml.m5.xlarge")
        instance_count = config.get("instance_count", 1)
        max_run_time = config.get("max_run_time", 3600)

        job_name = f"mbt-{algorithm}-{uuid.uuid4().hex[:8]}"
        self._current_job_name = job_name

        print(f"  Launching SageMaker training job: {job_name}")
        print(f"    Algorithm: {algorithm}")
        print(f"    Instance: {instance_type} x {instance_count}")

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

            estimator.fit({"train": s3_input_uri}, job_name=job_name, wait=True)

            print(f"  ✓ Training completed: {job_name}")

            model_data_uri = estimator.model_data

            return {
                "algorithm": algorithm,
                "job_name": job_name,
                "model_data_s3_uri": model_data_uri,
                "estimator": estimator,
            }

        except Exception as e:
            raise SageMakerTrainingError(
                f"SageMaker training job failed: {str(e)}"
            ) from e

    def _train_autopilot(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> dict:
        """Train using SageMaker Autopilot (AutoML).

        Autopilot automatically explores multiple algorithms and hyperparameters
        to find the best model — similar to H2O AutoML but running on SageMaker
        managed infrastructure.

        Args:
            X_train: Training features
            y_train: Training labels
            config: Autopilot configuration

        Returns:
            Dictionary with best candidate info, job name, and model artifacts
        """
        from sagemaker.automl.automlv2 import AutoMLV2
        from sagemaker.automl.automlv2 import AutoMLTabularConfig

        X_df = X_train.to_pandas()
        y_df = y_train.to_pandas()

        if len(y_df.columns) != 1:
            raise ValueError("y_train must have exactly one column")

        target_col = config["target_attribute"]

        # Combine X and y — Autopilot expects a single dataset with the target column
        train_df = X_df.copy()
        train_df[target_col] = y_df.iloc[:, 0].values

        print(f"  Preparing Autopilot training data: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"    Target column: {target_col}")

        # Upload to S3 (WITH header — Autopilot requires column names)
        try:
            s3_input_uri = upload_dataframe_to_s3(
                df=train_df,
                bucket=self._s3_bucket,
                prefix=f"{self._s3_prefix}autopilot-data/",
                session=self._session,
                include_header=True,
            )
            print(f"  ✓ Data uploaded to: {s3_input_uri}")
        except Exception as e:
            raise SageMakerTrainingError(
                f"Failed to upload Autopilot data to S3: {str(e)}"
            ) from e

        # Resolve problem type
        problem_type = config.get("autopilot_problem_type")
        if not problem_type:
            # Auto-map from MBT problem type if available in config context
            # Fall back to letting Autopilot infer it
            problem_type = None

        # Build Autopilot configuration
        tabular_config_kwargs = {
            "target_attribute_name": target_col,
        }
        if problem_type:
            tabular_config_kwargs["problem_type"] = problem_type

        tabular_config = AutoMLTabularConfig(**tabular_config_kwargs)

        # Timing constraints
        automl_kwargs = {
            "problem_config": tabular_config,
            "role": self._role_arn,
            "sagemaker_session": self._sagemaker_session,
            "output_path": f"s3://{self._s3_bucket}/{self._s3_prefix}autopilot/",
        }

        if "max_candidates" in config:
            automl_kwargs["max_candidates"] = config["max_candidates"]

        if "max_runtime_per_training_job_in_seconds" in config:
            automl_kwargs["max_runtime_per_training_job_in_seconds"] = (
                config["max_runtime_per_training_job_in_seconds"]
            )

        if "total_job_runtime_in_seconds" in config:
            automl_kwargs["total_job_runtime_in_seconds"] = (
                config["total_job_runtime_in_seconds"]
            )

        job_name = f"mbt-autopilot-{uuid.uuid4().hex[:8]}"
        self._current_job_name = job_name

        print(f"  Launching SageMaker Autopilot job: {job_name}")
        if problem_type:
            print(f"    Problem type: {problem_type}")
        else:
            print(f"    Problem type: Auto (inferred by Autopilot)")
        print(f"    Max candidates: {config.get('max_candidates', 'default')}")

        try:
            automl = AutoMLV2(**automl_kwargs)

            automl.fit(s3_input_uri, job_name=job_name, wait=True)

            best_candidate = automl.best_candidate()

            print(f"  ✓ Autopilot completed: {job_name}")
            print(f"    Best candidate: {best_candidate['CandidateName']}")

            return {
                "algorithm": "autopilot",
                "job_name": job_name,
                "best_candidate": best_candidate,
                "model_data_s3_uri": best_candidate.get("InferenceContainers", [{}])[0].get("ModelDataUrl", ""),
                "automl": automl,
            }

        except Exception as e:
            raise SageMakerTrainingError(
                f"SageMaker Autopilot job failed: {str(e)}"
            ) from e

    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions using trained SageMaker model.

        Uses SageMaker Batch Transform to generate predictions on the provided data.
        For Autopilot models, creates a Model from the best candidate's inference
        containers and runs Batch Transform.

        Args:
            model: Model dict from train() containing model info
            X: Features to predict on

        Returns:
            Predictions as numpy array
        """
        import pandas as pd

        X_df = X.to_pandas()

        algorithm = model.get("algorithm", "")

        # Upload inference data to S3
        try:
            s3_input_uri = upload_dataframe_to_s3(
                df=X_df,
                bucket=self._s3_bucket,
                prefix=f"{self._s3_prefix}inference-data/",
                session=self._session,
                include_header=(algorithm == "autopilot"),
            )
        except Exception as e:
            raise SageMakerTrainingError(
                f"Failed to upload inference data to S3: {str(e)}"
            ) from e

        s3_output_path = f"s3://{self._s3_bucket}/{self._s3_prefix}predictions/{uuid.uuid4().hex[:8]}/"

        try:
            if algorithm == "autopilot":
                automl = model.get("automl")
                best_candidate = model["best_candidate"]

                # Create a Model from the best candidate's inference containers
                inference_containers = best_candidate.get("InferenceContainers", [])
                if not inference_containers:
                    raise SageMakerTrainingError(
                        "Autopilot best candidate has no inference containers"
                    )

                sm_model = automl.create_model(
                    name=f"mbt-autopilot-model-{uuid.uuid4().hex[:8]}",
                    candidate=best_candidate,
                    sagemaker_session=self._sagemaker_session,
                )

                transformer = sm_model.transformer(
                    instance_count=1,
                    instance_type="ml.m5.xlarge",
                    output_path=s3_output_path,
                )
            else:
                estimator = model.get("estimator")
                if not estimator:
                    raise SageMakerTrainingError(
                        "Model dict missing 'estimator' reference. "
                        "Predict is only supported on models from the current session."
                    )
                transformer = estimator.transformer(
                    instance_count=1,
                    instance_type="ml.m5.xlarge",
                    output_path=s3_output_path,
                )

            print(f"  Launching Batch Transform job for predictions...")
            transformer.transform(
                s3_input_uri,
                content_type="text/csv",
                split_type="Line",
                wait=True,
            )
            print(f"  ✓ Batch Transform completed")

            # Download predictions from S3
            s3_client = self._session.client("s3")
            output_prefix = s3_output_path.replace(f"s3://{self._s3_bucket}/", "")

            response = s3_client.list_objects_v2(
                Bucket=self._s3_bucket, Prefix=output_prefix
            )
            if "Contents" not in response:
                raise SageMakerTrainingError("No prediction output found in S3")

            # Read the first output file
            output_key = response["Contents"][0]["Key"]
            obj = s3_client.get_object(Bucket=self._s3_bucket, Key=output_key)
            predictions_csv = obj["Body"].read().decode("utf-8")

            predictions = pd.read_csv(
                pd.io.common.StringIO(predictions_csv), header=None
            )
            return predictions.iloc[:, 0].values

        except SageMakerTrainingError:
            raise
        except Exception as e:
            raise SageMakerTrainingError(
                f"Batch Transform prediction failed: {str(e)}"
            ) from e

    def serialize(self, model: Any, path: str) -> None:
        """Save model metadata to disk.

        Stores job name, S3 URI, algorithm type, and for Autopilot models,
        the best candidate info needed to recreate the model for inference.

        Args:
            model: Model dict from train()
            path: Local path to save metadata
        """
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "algorithm": model.get("algorithm", "unknown"),
            "job_name": model["job_name"],
            "model_data_s3_uri": model["model_data_s3_uri"],
        }

        # For Autopilot, persist the best candidate info for later inference
        if model.get("algorithm") == "autopilot" and "best_candidate" in model:
            candidate = model["best_candidate"]
            metadata["best_candidate"] = {
                "CandidateName": candidate.get("CandidateName"),
                "InferenceContainers": candidate.get("InferenceContainers", []),
            }

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def deserialize(self, path: str) -> Any:
        """Load model metadata from disk.

        Args:
            path: Path to metadata file

        Returns:
            Model dict with job_name, model_data_s3_uri, algorithm, etc.
        """
        import json

        with open(path, "r") as f:
            return json.load(f)

    def get_training_metrics(self, model: Any) -> dict[str, float]:
        """Extract training metrics from SageMaker job.

        For built-in algorithms, reads FinalMetricDataList from the training job.
        For Autopilot, reads the best candidate's objective metric and inference
        container details.

        Args:
            model: Model dict from train()

        Returns:
            Dictionary of training metrics
        """
        job_name = model.get("job_name")
        if not job_name:
            return {}

        algorithm = model.get("algorithm", "")

        try:
            sm_client = self._session.client("sagemaker")

            if algorithm == "autopilot":
                return self._get_autopilot_metrics(sm_client, model)
            else:
                return self._get_builtin_metrics(sm_client, job_name)
        except Exception:
            return {}

    def _get_builtin_metrics(self, sm_client: Any, job_name: str) -> dict[str, float]:
        """Extract metrics from a built-in algorithm training job."""
        job_desc = sm_client.describe_training_job(TrainingJobName=job_name)

        metrics = {}
        if "FinalMetricDataList" in job_desc:
            for metric in job_desc["FinalMetricDataList"]:
                metrics[metric["MetricName"]] = float(metric["Value"])

        return metrics

    def _get_autopilot_metrics(self, sm_client: Any, model: dict) -> dict[str, float]:
        """Extract metrics from an Autopilot job's best candidate."""
        metrics = {}

        best_candidate = model.get("best_candidate", {})
        if not best_candidate:
            return metrics

        # Extract the objective metric from the best candidate
        objective_metric = best_candidate.get("FinalAutoMLJobObjectiveMetric", {})
        if objective_metric:
            metric_name = objective_metric.get("MetricName", "ObjectiveMetric")
            metric_value = objective_metric.get("Value")
            if metric_value is not None:
                metrics[metric_name] = float(metric_value)
                metrics["objective_type"] = 1.0 if objective_metric.get("Type") == "Maximize" else 0.0

        # Include candidate name as metadata
        candidate_name = best_candidate.get("CandidateName", "")
        if candidate_name:
            metrics["_candidate_name"] = hash(candidate_name) % 1000000  # numeric summary

        return metrics
