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


class FeatureTableConfig(BaseModel):
    """Feature table specification for multi-table joins."""

    table: str = Field(..., description="Feature table name")
    join_key: list[str] = Field(
        default=["customer_id", "snapshot_date"],
        description="Columns to join on (typically primary_key + partition_key)"
    )
    join_type: str = Field(
        default="left",
        description="Join type: inner, left, right, outer"
    )
    fan_out_check: bool = Field(
        default=True,
        description="Raise error if join increases row count unexpectedly"
    )


class WindowsSpec(BaseModel):
    """Window specifications for train/test splitting."""

    test_lookback_units: int = Field(
        ...,
        description="How many units of data to use for test set (from latest available)"
    )
    train_gap_units: int = Field(
        default=0,
        description="Gap between train and test sets (0 = no gap)"
    )
    train_lookback_units: int = Field(
        ...,
        description="How many units of historical data for training"
    )

    # For absolute mode (optional)
    train_start_date: Optional[str] = Field(None, description="Fixed train start (YYYY-MM-DD)")
    train_end_date: Optional[str] = Field(None, description="Fixed train end (YYYY-MM-DD)")
    test_start_date: Optional[str] = Field(None, description="Fixed test start (YYYY-MM-DD)")
    test_end_date: Optional[str] = Field(None, description="Fixed test end (YYYY-MM-DD)")


class DataWindowsConfig(BaseModel):
    """Temporal windowing configuration for train/test splitting.

    Supports two modes:
    - relative: Windows shift with execution_date (production use)
    - absolute: Fixed date ranges (reproducible experiments)
    """

    logic: str = Field(
        default="relative",
        description="Window mode: 'relative' or 'absolute'"
    )
    unit_type: str = Field(
        default="months",
        description="Time unit: days, weeks, months, quarters, years"
    )
    windows: WindowsSpec = Field(
        ...,
        description="Train/test window specifications"
    )


class DataSourceConfig(BaseModel):
    """Data source specification - where to load data from."""

    label_table: str = Field(..., description="Primary table with labels")
    feature_tables: Optional[list[FeatureTableConfig]] = Field(
        default=None,
        description="Additional feature tables to join"
    )
    data_windows: Optional[DataWindowsConfig] = Field(
        default=None,
        description="Temporal windowing configuration"
    )


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
    temporal_analysis: Optional[dict] = Field(
        default=None,
        description="Temporal drift analysis. Config: {enabled: true, partition_key: 'snapshot_date', metrics: ['roc_auc', 'psi']}"
    )


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration."""

    remove_high_missing: Optional[dict] = Field(
        default=None,
        description="Remove columns with high missing rate. Config: {enabled: true, threshold: 0.95}"
    )
    remove_constant: Optional[dict] = Field(
        default=None,
        description="Remove constant features. Config: {enabled: true}"
    )
    imputation: Optional[dict] = Field(
        default=None,
        description="Missing value imputation. Config: {enabled: true, strategy: 'median'|'mean'|'mode'}"
    )


class FeatureSelectionMethodConfig(BaseModel):
    """Single feature selection method configuration."""

    name: str = Field(..., description="Method name: variance_threshold, correlation, mutual_info, lgbm_importance")
    threshold: Optional[float] = Field(None, description="Threshold value (method-specific)")
    k: Optional[int] = Field(None, description="Number of features to keep (for mutual_info)")


class FeatureSelectionConfig(BaseModel):
    """Feature selection pipeline configuration."""

    enabled: bool = Field(default=False, description="Enable feature selection")
    methods: list[FeatureSelectionMethodConfig] = Field(default_factory=list, description="Selection methods to apply in order")


class TrainingConfig(BaseModel):
    """Complete training pipeline configuration."""

    data_source: DataSourceConfig
    schema: SchemaConfig
    preprocessing: Optional[PreprocessingConfig] = Field(default=None, description="Data preprocessing steps")
    feature_selection: Optional[FeatureSelectionConfig] = Field(default=None, description="Feature selection configuration")
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
