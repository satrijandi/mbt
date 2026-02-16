# mbt-sagemaker

AWS SageMaker framework adapter for MBT (Model Build Tool).

## Overview

This adapter enables MBT pipelines to run training jobs on AWS SageMaker infrastructure using built-in algorithms like XGBoost, LinearLearner, and more.

## Installation

```bash
pip install -e mbt-sagemaker/
```

Or with uv:

```bash
uv pip install -e mbt-sagemaker/
```

## Configuration

### profiles.yaml

```yaml
outputs:
  prod:
    sagemaker_connection:
      region: us-east-1
      role_arn: arn:aws:iam::ACCOUNT:role/SageMakerRole
      s3_bucket: your-bucket-name
      s3_prefix: mbt-models/
```

### Pipeline YAML

```yaml
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
```

## Supported Algorithms

- **xgboost**: Gradient boosting for classification and regression
- **linear-learner**: Linear models with SGD optimization
- **factorization-machines**: Recommendation and click prediction
- **knn**: K-nearest neighbors

## Requirements

- AWS credentials configured (via environment, ~/.aws/credentials, or IAM role)
- SageMaker execution role with appropriate permissions
- S3 bucket for training data and model artifacts

## Development

```bash
# Install with dev dependencies
uv pip install -e "mbt-sagemaker/[dev]"

# Run tests
cd mbt-sagemaker/
pytest tests/ -v
```

## License

Same as MBT core.
