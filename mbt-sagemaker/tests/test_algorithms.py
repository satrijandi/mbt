"""Tests for SageMaker algorithm specifications and validation."""

import pytest

from mbt_sagemaker.algorithms import (
    SUPPORTED_ALGORITHMS,
    AUTOPILOT_PROBLEM_TYPES,
    MBT_TO_AUTOPILOT_PROBLEM_TYPE,
    get_builtin_algorithm_spec,
    validate_algorithm_config,
)


class TestSupportedAlgorithms:
    def test_autopilot_in_supported_algorithms(self):
        assert "autopilot" in SUPPORTED_ALGORITHMS

    def test_builtin_algorithms_present(self):
        for algo in ["xgboost", "linear-learner", "factorization-machines", "knn"]:
            assert algo in SUPPORTED_ALGORITHMS


class TestAutopilotProblemTypeMapping:
    def test_binary_classification_maps(self):
        assert MBT_TO_AUTOPILOT_PROBLEM_TYPE["binary_classification"] == "BinaryClassification"

    def test_multiclass_classification_maps(self):
        assert MBT_TO_AUTOPILOT_PROBLEM_TYPE["multiclass_classification"] == "MulticlassClassification"

    def test_regression_maps(self):
        assert MBT_TO_AUTOPILOT_PROBLEM_TYPE["regression"] == "Regression"


class TestAutopilotValidation:
    def test_valid_autopilot_config(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "churn",
            "max_candidates": 10,
        }
        # Should not raise
        validate_algorithm_config("autopilot", config, "binary_classification")

    def test_missing_target_attribute(self):
        config = {
            "algorithm": "autopilot",
            "max_candidates": 10,
        }
        with pytest.raises(ValueError, match="target_attribute"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_invalid_problem_type(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
        }
        with pytest.raises(ValueError, match="Unsupported problem_type"):
            validate_algorithm_config("autopilot", config, "unsupported_type")

    def test_invalid_autopilot_problem_type(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "autopilot_problem_type": "InvalidType",
        }
        with pytest.raises(ValueError, match="Invalid autopilot_problem_type"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_valid_explicit_autopilot_problem_type(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "autopilot_problem_type": "BinaryClassification",
        }
        validate_algorithm_config("autopilot", config, "binary_classification")

    def test_invalid_max_candidates_zero(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "max_candidates": 0,
        }
        with pytest.raises(ValueError, match="max_candidates"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_invalid_max_candidates_negative(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "max_candidates": -5,
        }
        with pytest.raises(ValueError, match="max_candidates"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_invalid_max_candidates_type(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "max_candidates": "ten",
        }
        with pytest.raises(ValueError, match="max_candidates"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_invalid_max_runtime_per_job(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "max_runtime_per_training_job_in_seconds": -1,
        }
        with pytest.raises(ValueError, match="max_runtime_per_training_job_in_seconds"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_invalid_total_job_runtime(self):
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
            "total_job_runtime_in_seconds": 0,
        }
        with pytest.raises(ValueError, match="total_job_runtime_in_seconds"):
            validate_algorithm_config("autopilot", config, "binary_classification")

    def test_all_valid_mbt_problem_types(self):
        """Autopilot validation should pass for all mapped MBT problem types."""
        config = {
            "algorithm": "autopilot",
            "target_attribute": "target",
        }
        for problem_type in MBT_TO_AUTOPILOT_PROBLEM_TYPE:
            validate_algorithm_config("autopilot", config, problem_type)


class TestBuiltinAlgorithmSpec:
    def test_autopilot_not_in_algorithm_images(self):
        """Autopilot doesn't use image URIs, so get_builtin_algorithm_spec should fail."""
        with pytest.raises(ValueError, match="not supported"):
            get_builtin_algorithm_spec("autopilot", "us-east-1")
