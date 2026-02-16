"""Custom exceptions for SageMaker adapter."""


class SageMakerError(Exception):
    """Base exception for SageMaker adapter errors."""
    pass


class SageMakerSetupError(SageMakerError):
    """Raised when SageMaker setup fails."""
    pass


class SageMakerTrainingError(SageMakerError):
    """Raised when SageMaker training job fails."""
    pass
