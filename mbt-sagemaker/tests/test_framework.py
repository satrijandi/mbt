"""Tests for SageMaker framework adapter — Autopilot and built-in algorithm support."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from mbt_sagemaker.framework import SageMakerFramework
from mbt_sagemaker.exceptions import SageMakerTrainingError


class MockMBTFrame:
    """Mock MBTFrame for testing."""

    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


@pytest.fixture
def framework():
    """Create a SageMakerFramework with mocked AWS session state."""
    fw = SageMakerFramework()
    fw._session = MagicMock()
    fw._sagemaker_session = MagicMock()
    fw._role_arn = "arn:aws:iam::123456789:role/TestRole"
    fw._s3_bucket = "test-bucket"
    fw._s3_prefix = "mbt-test/"
    fw._region = "us-east-1"
    return fw


class TestValidateConfig:
    def test_missing_algorithm(self):
        fw = SageMakerFramework()
        with pytest.raises(ValueError, match="must specify 'algorithm'"):
            fw.validate_config({}, "binary_classification")

    def test_unsupported_algorithm(self):
        fw = SageMakerFramework()
        with pytest.raises(ValueError, match="Unsupported SageMaker algorithm"):
            fw.validate_config({"algorithm": "nonexistent"}, "binary_classification")

    def test_autopilot_valid_config(self):
        fw = SageMakerFramework()
        config = {
            "algorithm": "autopilot",
            "target_attribute": "churn",
            "max_candidates": 20,
        }
        fw.validate_config(config, "binary_classification")

    def test_autopilot_missing_target(self):
        fw = SageMakerFramework()
        config = {"algorithm": "autopilot"}
        with pytest.raises(ValueError, match="target_attribute"):
            fw.validate_config(config, "binary_classification")

    def test_invalid_instance_type(self):
        fw = SageMakerFramework()
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "instance_type": "invalid.type",
        }
        with pytest.raises(ValueError, match="Invalid instance_type"):
            fw.validate_config(config, "binary_classification")

    def test_xgboost_still_works(self):
        fw = SageMakerFramework()
        config = {
            "algorithm": "xgboost",
            "hyperparameters": {"objective": "binary:logistic"},
        }
        fw.validate_config(config, "binary_classification")


class TestTrainRouting:
    def test_autopilot_routes_to_train_autopilot(self, framework):
        """Verify train() delegates to _train_autopilot for algorithm=autopilot."""
        framework._train_autopilot = MagicMock(return_value={"algorithm": "autopilot"})
        framework._train_builtin = MagicMock()

        X = MagicMock()
        y = MagicMock()
        config = {"algorithm": "autopilot", "target_attribute": "churn"}

        framework.train(X, y, config)

        framework._train_autopilot.assert_called_once_with(X, y, config)
        framework._train_builtin.assert_not_called()

    def test_builtin_routes_to_train_builtin(self, framework):
        """Verify train() delegates to _train_builtin for built-in algorithms."""
        framework._train_autopilot = MagicMock()
        framework._train_builtin = MagicMock(return_value={"algorithm": "xgboost"})

        X = MagicMock()
        y = MagicMock()
        config = {"algorithm": "xgboost"}

        framework.train(X, y, config)

        framework._train_builtin.assert_called_once_with(X, y, config)
        framework._train_autopilot.assert_not_called()


class TestTrainAutopilot:
    @patch("mbt_sagemaker.framework.upload_dataframe_to_s3")
    def test_autopilot_uploads_with_header(self, mock_upload, framework):
        """Autopilot data upload must include CSV headers."""
        import pandas as pd

        mock_upload.return_value = "s3://test-bucket/data.csv"

        X_df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
        y_df = pd.DataFrame({"churn": [0, 1]})

        config = {
            "algorithm": "autopilot",
            "target_attribute": "churn",
        }

        with patch("mbt_sagemaker.framework.AutoMLV2", create=True) as MockAutoML, \
             patch("mbt_sagemaker.framework.AutoMLTabularConfig", create=True):
            # Need to mock the sagemaker.automl imports inside _train_autopilot
            import sys
            mock_automl_module = MagicMock()
            mock_automl_v2 = MagicMock()
            mock_automl_v2_instance = MagicMock()
            mock_automl_v2_instance.best_candidate.return_value = {
                "CandidateName": "test-candidate",
                "InferenceContainers": [{"ModelDataUrl": "s3://bucket/model.tar.gz"}],
            }
            mock_automl_v2.return_value = mock_automl_v2_instance
            mock_automl_module.AutoMLV2 = mock_automl_v2
            mock_automl_module.AutoMLTabularConfig = MagicMock()
            sys.modules["sagemaker.automl.automlv2"] = mock_automl_module

            try:
                result = framework._train_autopilot(
                    MockMBTFrame(X_df), MockMBTFrame(y_df), config
                )
            finally:
                del sys.modules["sagemaker.automl.automlv2"]

        # Verify header was included
        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args
        assert call_kwargs.kwargs.get("include_header") is True or \
               (len(call_kwargs.args) > 4 and call_kwargs.args[4] is True) or \
               call_kwargs[1].get("include_header") is True

    @patch("mbt_sagemaker.framework.upload_dataframe_to_s3")
    def test_autopilot_combines_x_and_y(self, mock_upload, framework):
        """Autopilot should combine X and y into a single DataFrame with target column."""
        import pandas as pd

        mock_upload.return_value = "s3://test-bucket/data.csv"

        X_df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
        y_df = pd.DataFrame({"label": [0, 1]})

        config = {
            "algorithm": "autopilot",
            "target_attribute": "churn",
        }

        import sys
        mock_automl_module = MagicMock()
        mock_automl_v2_instance = MagicMock()
        mock_automl_v2_instance.best_candidate.return_value = {
            "CandidateName": "test-candidate",
            "InferenceContainers": [{"ModelDataUrl": "s3://bucket/model.tar.gz"}],
        }
        mock_automl_module.AutoMLV2.return_value = mock_automl_v2_instance
        mock_automl_module.AutoMLTabularConfig = MagicMock()
        sys.modules["sagemaker.automl.automlv2"] = mock_automl_module

        try:
            framework._train_autopilot(
                MockMBTFrame(X_df), MockMBTFrame(y_df), config
            )
        finally:
            del sys.modules["sagemaker.automl.automlv2"]

        # Check the uploaded DataFrame has the target column
        uploaded_df = mock_upload.call_args[1]["df"] if "df" in mock_upload.call_args[1] else mock_upload.call_args[0][0]
        assert "churn" in uploaded_df.columns
        assert "feat1" in uploaded_df.columns
        assert "feat2" in uploaded_df.columns
        assert len(uploaded_df) == 2

    def test_autopilot_rejects_multi_column_target(self, framework):
        """Should raise ValueError if y has multiple columns."""
        import pandas as pd

        X_df = pd.DataFrame({"feat1": [1]})
        y_df = pd.DataFrame({"a": [1], "b": [2]})

        config = {"algorithm": "autopilot", "target_attribute": "target"}

        with pytest.raises(ValueError, match="exactly one column"):
            framework._train_autopilot(
                MockMBTFrame(X_df), MockMBTFrame(y_df), config
            )


class TestSerializeDeserialize:
    def test_serialize_autopilot_model(self):
        fw = SageMakerFramework()

        model = {
            "algorithm": "autopilot",
            "job_name": "mbt-autopilot-abc12345",
            "model_data_s3_uri": "s3://bucket/model.tar.gz",
            "best_candidate": {
                "CandidateName": "best-candidate-001",
                "InferenceContainers": [
                    {"ModelDataUrl": "s3://bucket/model.tar.gz", "Image": "123.dkr.ecr.us-east-1.amazonaws.com/image:1"}
                ],
                "FinalAutoMLJobObjectiveMetric": {"MetricName": "F1", "Value": 0.95},
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            fw.serialize(model, path)

            with open(path) as f:
                saved = json.load(f)

            assert saved["algorithm"] == "autopilot"
            assert saved["job_name"] == "mbt-autopilot-abc12345"
            assert saved["model_data_s3_uri"] == "s3://bucket/model.tar.gz"
            assert "best_candidate" in saved
            assert saved["best_candidate"]["CandidateName"] == "best-candidate-001"
            assert len(saved["best_candidate"]["InferenceContainers"]) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_serialize_builtin_model_no_candidate(self):
        fw = SageMakerFramework()

        model = {
            "algorithm": "xgboost",
            "job_name": "mbt-xgboost-abc12345",
            "model_data_s3_uri": "s3://bucket/model.tar.gz",
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            fw.serialize(model, path)

            with open(path) as f:
                saved = json.load(f)

            assert saved["algorithm"] == "xgboost"
            assert "best_candidate" not in saved
        finally:
            Path(path).unlink(missing_ok=True)

    def test_deserialize_roundtrip(self):
        fw = SageMakerFramework()

        model = {
            "algorithm": "autopilot",
            "job_name": "mbt-autopilot-abc12345",
            "model_data_s3_uri": "s3://bucket/model.tar.gz",
            "best_candidate": {
                "CandidateName": "best-candidate-001",
                "InferenceContainers": [],
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            fw.serialize(model, path)
            loaded = fw.deserialize(path)

            assert loaded["algorithm"] == "autopilot"
            assert loaded["job_name"] == model["job_name"]
            assert loaded["best_candidate"]["CandidateName"] == "best-candidate-001"
        finally:
            Path(path).unlink(missing_ok=True)


class TestGetTrainingMetrics:
    def test_autopilot_metrics_extraction(self, framework):
        model = {
            "algorithm": "autopilot",
            "job_name": "mbt-autopilot-test",
            "best_candidate": {
                "CandidateName": "best-xgboost-001",
                "FinalAutoMLJobObjectiveMetric": {
                    "MetricName": "F1",
                    "Value": 0.92,
                    "Type": "Maximize",
                },
            },
        }

        metrics = framework.get_training_metrics(model)

        assert metrics["F1"] == 0.92
        assert metrics["objective_type"] == 1.0  # Maximize

    def test_autopilot_minimize_metric(self, framework):
        model = {
            "algorithm": "autopilot",
            "job_name": "mbt-autopilot-test",
            "best_candidate": {
                "CandidateName": "best-linear-001",
                "FinalAutoMLJobObjectiveMetric": {
                    "MetricName": "MSE",
                    "Value": 0.05,
                    "Type": "Minimize",
                },
            },
        }

        metrics = framework.get_training_metrics(model)

        assert metrics["MSE"] == 0.05
        assert metrics["objective_type"] == 0.0  # Minimize

    def test_autopilot_no_candidate_returns_empty(self, framework):
        model = {
            "algorithm": "autopilot",
            "job_name": "mbt-autopilot-test",
        }

        metrics = framework.get_training_metrics(model)
        assert metrics == {}

    def test_builtin_metrics_via_describe_job(self, framework):
        sm_client = MagicMock()
        sm_client.describe_training_job.return_value = {
            "FinalMetricDataList": [
                {"MetricName": "train:rmse", "Value": 0.123},
                {"MetricName": "validation:rmse", "Value": 0.145},
            ]
        }
        framework._session.client.return_value = sm_client

        model = {
            "algorithm": "xgboost",
            "job_name": "mbt-xgboost-test",
        }

        metrics = framework.get_training_metrics(model)

        assert metrics["train:rmse"] == 0.123
        assert metrics["validation:rmse"] == 0.145

    def test_no_job_name_returns_empty(self, framework):
        metrics = framework.get_training_metrics({})
        assert metrics == {}
