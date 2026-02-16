"""Manifest models - the compiled executable representation of a pipeline.

The manifest is the contract between the compiler and the runner. It's a complete,
serializable execution plan with all configuration resolved.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ManifestMetadata(BaseModel):
    """Metadata about the compiled manifest."""

    mbt_version: str = Field(..., description="MBT version that compiled this manifest")
    schema_version: int = Field(..., description="Pipeline schema version")
    generated_at: str = Field(..., description="ISO timestamp of compilation")
    pipeline_name: str = Field(..., description="Unique pipeline name")
    pipeline_type: str = Field(..., description="training or serving")
    target: str = Field("dev", description="Target environment (dev, staging, prod)")
    problem_type: str = Field(..., description="ML problem type")


class StepDefinition(BaseModel):
    """Definition of a single pipeline step."""

    plugin: str = Field(..., description="Python import path to step class (e.g., mbt.steps.load_data:LoadDataStep)")
    config: dict[str, Any] = Field(default_factory=dict, description="Step-specific configuration")
    resources: dict[str, str] = Field(default_factory=dict, description="Resource limits (cpu, memory)")
    inputs: list[str] = Field(default_factory=list, description="Input artifact names this step consumes")
    outputs: list[str] = Field(default_factory=list, description="Output artifact names this step produces")
    depends_on: list[str] = Field(default_factory=list, description="Step names this step depends on")
    idempotent: bool = Field(True, description="Whether step can be safely retried with same inputs")


class DAGDefinition(BaseModel):
    """DAG structure and execution order."""

    parent_map: dict[str, list[str]] = Field(..., description="step_name -> [parent_step_names]")
    execution_batches: list[list[str]] = Field(..., description="Steps grouped by execution batch (topologically sorted)")


class Manifest(BaseModel):
    """Complete compiled manifest - ready for execution."""

    metadata: ManifestMetadata
    steps: dict[str, StepDefinition] = Field(..., description="step_name -> StepDefinition")
    dag: DAGDefinition

    class Config:
        extra = "allow"  # Allow additional fields for extensibility
