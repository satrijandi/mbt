"""Pipeline YAML schema - Pydantic models for validation and type safety.

This defines the data scientist-facing API. Schema versioning allows evolution
without breaking existing pipelines.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Project Metadata
# =============================================================================


class ProjectConfig(BaseModel):
    """Project-level metadata and configuration."""

    name: str = Field(..., description="Unique pipeline name")
    experiment_name: Optional[str] = Field(None, description="MLflow experiment name")
    owner: Optional[str] = Field(None, description="Team or individual owner")
    problem_type: Literal["binary_classification", "multiclass_classification", "regression"] = (
        Field(..., description="ML problem type - drives evaluation metrics and strategies")
    )
    tags: list[str] = Field(default_factory=list, description="Tags for filtering and organization")


# =============================================================================
# Training Configuration
# =============================================================================


class DataSourceConfig(BaseModel):
    """Data source specification - where to load data from."""

    label_table: str = Field(..., description="Primary table with labels")
    # Phase 4 additions: feature_tables, data_windows
    # For Phase 1: just load from single table


class TargetConfig(BaseModel):
    """Target/label column specification."""

    label_column: str = Field(..., description="Name of the target/label column")
    classes: Optional[list] = Field(None, description="Class labels for classification")
    positive_class: Optional[Any] = Field(None, description="Positive class for binary classification")


class IdentifiersConfig(BaseModel):
    """Column identifiers - keys and partitions."""

    primary_key: str = Field(..., description="Primary key column (e.g., customer_id)")
    partition_key: Optional[str] = Field(None, description="Partition/date column (e.g., snapshot_date)")


class SchemaConfig(BaseModel):
    """Dataset schema specification."""

    target: TargetConfig
    identifiers: IdentifiersConfig
    ignored_columns: list[str] = Field(default_factory=list, description="Columns to exclude from training")


class ModelTrainingConfig(BaseModel):
    """Model training configuration."""

    framework: str = Field("sklearn", description="ML framework (sklearn, h2o_automl, xgboost, etc.)")
    config: dict[str, Any] = Field(default_factory=dict, description="Framework-specific config (pass-through)")
    resources: Optional[dict[str, str]] = Field(None, description="Resource limits (cpu, memory)")


class EvaluationConfig(BaseModel):
    """Model evaluation configuration."""

    primary_metric: str = Field("roc_auc", description="Primary metric for model selection")
    additional_metrics: list[str] = Field(default_factory=list, description="Additional metrics to compute")
    generate_plots: bool = Field(True, description="Generate evaluation plots")


class TrainingConfig(BaseModel):
    """Complete training pipeline configuration."""

    data_source: DataSourceConfig
    schema: SchemaConfig
    model_training: ModelTrainingConfig
    evaluation: EvaluationConfig


# =============================================================================
# Serving Configuration (Phase 5)
# =============================================================================


class ModelSourceConfig(BaseModel):
    """Model source specification for serving pipelines."""

    registry: str = Field("mlflow", description="Model registry to load from (mlflow, etc.)")
    run_id: str = Field(..., description="Model run ID to load (MLflow run_id, etc.)")
    artifact_snapshot: bool = Field(False, description="Fetch artifacts at compile time")
    fallback_run_id: Optional[str] = Field(None, description="Fallback run_id if primary fails")


class ServingDataSourceConfig(BaseModel):
    """Data source for serving/scoring."""

    scoring_table: str = Field(..., description="Table with data to score")
    # Phase 5: Simple table name
    # Later: data_windows for lookback periods


class OutputConfig(BaseModel):
    """Output configuration for predictions."""

    destination: str = Field("local_file", description="Output destination type (local_file, database, etc.)")
    table: Optional[str] = Field(None, description="Output table/file name")
    path: Optional[str] = Field(None, description="Output file path")
    columns: Optional[dict[str, Any]] = Field(None, description="Output column configuration")
    include_probabilities: bool = Field(True, description="Include prediction probabilities")


class DeploymentConfig(BaseModel):
    """Deployment configuration."""

    mode: Literal["batch", "realtime", "streaming"] = Field("batch", description="Deployment mode")
    cadence: Optional[str] = Field(None, description="Batch cadence (daily, hourly, etc.)")


class ServingConfig(BaseModel):
    """Complete serving pipeline configuration."""

    model_source: ModelSourceConfig
    data_source: ServingDataSourceConfig
    output: OutputConfig


# =============================================================================
# Top-Level Pipeline Schema
# =============================================================================


class PipelineConfig(BaseModel):
    """Complete pipeline YAML schema."""

    schema_version: int = Field(1, description="Schema version for migrations")
    project: ProjectConfig
    training: Optional[TrainingConfig] = Field(None, description="Training pipeline configuration")
    serving: Optional[ServingConfig] = Field(None, description="Serving pipeline configuration")
    deployment: Optional[DeploymentConfig] = Field(None, description="Deployment configuration")

    class Config:
        extra = "forbid"  # Fail on unknown fields to catch typos
