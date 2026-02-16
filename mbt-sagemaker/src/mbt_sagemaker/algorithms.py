"""Built-in SageMaker algorithm specifications and validation."""

SUPPORTED_ALGORITHMS = [
    "xgboost",
    "linear-learner",
    "factorization-machines",
    "knn",
]

# Algorithm image URIs by region
# Note: These are specific to SageMaker built-in algorithms
ALGORITHM_IMAGES = {
    "xgboost": {
        "us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
        "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1",
        "eu-west-1": "685385470294.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost:1.5-1",
    },
    "linear-learner": {
        "us-east-1": "382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:1",
        "us-west-2": "174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1",
        "eu-west-1": "438346466558.dkr.ecr.eu-west-1.amazonaws.com/linear-learner:1",
    },
    "factorization-machines": {
        "us-east-1": "382416733822.dkr.ecr.us-east-1.amazonaws.com/factorization-machines:1",
        "us-west-2": "174872318107.dkr.ecr.us-west-2.amazonaws.com/factorization-machines:1",
        "eu-west-1": "438346466558.dkr.ecr.eu-west-1.amazonaws.com/factorization-machines:1",
    },
    "knn": {
        "us-east-1": "382416733822.dkr.ecr.us-east-1.amazonaws.com/knn:1",
        "us-west-2": "174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1",
        "eu-west-1": "438346466558.dkr.ecr.eu-west-1.amazonaws.com/knn:1",
    },
}


def get_builtin_algorithm_spec(algorithm: str, region: str) -> dict:
    """Get algorithm specification including image URI.

    Args:
        algorithm: Algorithm name (e.g., "xgboost")
        region: AWS region (e.g., "us-east-1")

    Returns:
        Dict with image_uri and algorithm name

    Raises:
        ValueError: If algorithm or region not supported
    """
    if algorithm not in ALGORITHM_IMAGES:
        raise ValueError(
            f"Algorithm '{algorithm}' not supported. "
            f"Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS)}"
        )

    region_images = ALGORITHM_IMAGES[algorithm]
    if region not in region_images:
        raise ValueError(
            f"Algorithm '{algorithm}' not available in region '{region}'. "
            f"Available regions: {', '.join(region_images.keys())}"
        )

    return {
        "image_uri": region_images[region],
        "algorithm": algorithm,
    }


def validate_algorithm_config(algorithm: str, config: dict, problem_type: str):
    """Validate algorithm-specific configuration.

    Args:
        algorithm: Algorithm name
        config: Full framework config
        problem_type: ML problem type (binary_classification, etc.)

    Raises:
        ValueError: If configuration invalid for algorithm
    """
    if algorithm == "xgboost":
        _validate_xgboost_config(config, problem_type)
    elif algorithm == "linear-learner":
        _validate_linear_learner_config(config, problem_type)
    # Add more algorithm-specific validation as needed


def _validate_xgboost_config(config: dict, problem_type: str):
    """Validate XGBoost-specific configuration."""
    hyperparams = config.get("hyperparameters", {})

    if "objective" in hyperparams:
        objective = hyperparams["objective"]

        # Validate objective matches problem type
        if problem_type == "binary_classification":
            if not objective.startswith("binary:"):
                raise ValueError(
                    f"XGBoost objective '{objective}' invalid for binary classification. "
                    "Use binary:logistic or binary:hinge"
                )
        elif problem_type == "multiclass_classification":
            if not objective.startswith("multi:"):
                raise ValueError(
                    f"XGBoost objective '{objective}' invalid for multiclass classification. "
                    "Use multi:softmax or multi:softprob"
                )
        elif problem_type == "regression":
            if not objective.startswith("reg:"):
                raise ValueError(
                    f"XGBoost objective '{objective}' invalid for regression. "
                    "Use reg:squarederror or reg:logistic"
                )


def _validate_linear_learner_config(config: dict, problem_type: str):
    """Validate Linear Learner-specific configuration."""
    hyperparams = config.get("hyperparameters", {})

    if "predictor_type" in hyperparams:
        predictor = hyperparams["predictor_type"]
        valid_types = ["binary_classifier", "multiclass_classifier", "regressor"]

        if predictor not in valid_types:
            raise ValueError(
                f"Invalid predictor_type: '{predictor}'. "
                f"Valid types: {', '.join(valid_types)}"
            )

        # Validate predictor type matches problem type
        if problem_type == "binary_classification" and predictor != "binary_classifier":
            raise ValueError(
                f"predictor_type '{predictor}' doesn't match problem_type '{problem_type}'"
            )
        elif problem_type == "multiclass_classification" and predictor != "multiclass_classifier":
            raise ValueError(
                f"predictor_type '{predictor}' doesn't match problem_type '{problem_type}'"
            )
        elif problem_type == "regression" and predictor != "regressor":
            raise ValueError(
                f"predictor_type '{predictor}' doesn't match problem_type '{problem_type}'"
            )
