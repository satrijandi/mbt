"""DAG builder - constructs execution graph from pipeline configuration.

For Phase 1: Simple linear DAG with hardcoded steps.
For Phase 4: Conditional step registry based on enabled YAML sections.
"""

import graphlib
from typing import Any
from mbt.core.manifest import DAGDefinition, StepDefinition


class DAGBuilder:
    """Builds DAG from pipeline configuration."""

    def __init__(self):
        pass

    def build_training_dag(self, pipeline_config: dict[str, Any]) -> tuple[dict[str, StepDefinition], DAGDefinition]:
        """Build training pipeline DAG.

        For Phase 1: Hardcoded linear DAG: load_data -> split_data -> train_model -> evaluate

        Returns:
            (steps_dict, dag_definition)
        """
        # Hardcoded steps for Phase 1
        steps = {}

        # Step 1: load_data
        steps["load_data"] = StepDefinition(
            plugin="mbt.steps.load_data:LoadDataStep",
            config={
                "label_table": pipeline_config["training"]["data_source"]["label_table"],
                "schema": pipeline_config["training"]["schema"],
            },
            inputs=[],
            outputs=["raw_data"],
            depends_on=[],
            idempotent=False,  # Data source may change
        )

        # Step 2: split_data
        steps["split_data"] = StepDefinition(
            plugin="mbt.steps.split_data:SplitDataStep",
            config={
                "test_size": 0.2,
                "stratify": True,
                "target_column": pipeline_config["training"]["schema"]["target"]["label_column"],
            },
            inputs=["raw_data"],
            outputs=["train_set", "test_set"],
            depends_on=["load_data"],
            idempotent=True,
        )

        # Step 3: train_model
        resources = pipeline_config["training"]["model_training"].get("resources")
        if resources is None:
            resources = {}

        steps["train_model"] = StepDefinition(
            plugin="mbt.steps.train_model:TrainModelStep",
            config={
                "framework": pipeline_config["training"]["model_training"]["framework"],
                "framework_config": pipeline_config["training"]["model_training"]["config"],
                "target_column": pipeline_config["training"]["schema"]["target"]["label_column"],
                "problem_type": pipeline_config["project"]["problem_type"],
                "schema": pipeline_config["training"]["schema"],  # Pass schema for column filtering
            },
            resources=resources,
            inputs=["train_set"],
            outputs=["model", "train_metrics"],
            depends_on=["split_data"],
            idempotent=False,  # Training is stochastic
        )

        # Step 4: evaluate
        steps["evaluate"] = StepDefinition(
            plugin="mbt.steps.evaluate:EvaluateStep",
            config={
                "primary_metric": pipeline_config["training"]["evaluation"]["primary_metric"],
                "additional_metrics": pipeline_config["training"]["evaluation"]["additional_metrics"],
                "problem_type": pipeline_config["project"]["problem_type"],
                "target_column": pipeline_config["training"]["schema"]["target"]["label_column"],
                "schema": pipeline_config["training"]["schema"],  # Pass schema for column filtering
                "framework": pipeline_config["training"]["model_training"]["framework"],
            },
            inputs=["model", "test_set"],
            outputs=["eval_metrics", "eval_plots"],
            depends_on=["train_model"],
            idempotent=True,
        )

        # Step 5: log_run (Phase 2: Log to model registry)
        steps["log_run"] = StepDefinition(
            plugin="mbt.steps.log_run:LogRunStep",
            config={
                "registry": "mlflow",  # Default to MLflow
                "pipeline_name": pipeline_config["project"]["name"],
                "framework": pipeline_config["training"]["model_training"]["framework"],
                "framework_config": pipeline_config["training"]["model_training"]["config"],
                "problem_type": pipeline_config["project"]["problem_type"],
                "target_column": pipeline_config["training"]["schema"]["target"]["label_column"],
                "tags": pipeline_config["project"].get("tags", {}),
            },
            inputs=["model", "train_metrics", "eval_metrics"],
            outputs=["run_id"],
            depends_on=["evaluate"],
            idempotent=False,  # Each run gets a new ID
        )

        # Build DAG structure
        dag = self._build_dag_structure(steps)

        return steps, dag

    def build_serving_dag(self, pipeline_config: dict[str, Any]) -> tuple[dict[str, StepDefinition], DAGDefinition]:
        """Build serving pipeline DAG.

        Serving pipeline: load_scoring_data → load_model → apply_transforms → predict → publish

        Returns:
            (steps_dict, dag_definition)
        """
        steps = {}

        # Get serving configuration
        serving_config = pipeline_config.get("serving", {})
        if not serving_config:
            raise ValueError("Serving configuration not found in pipeline YAML")

        # Step 1: load_scoring_data
        steps["load_scoring_data"] = StepDefinition(
            plugin="mbt.steps.load_scoring_data:LoadScoringDataStep",
            config={
                "scoring_table": serving_config["data_source"]["scoring_table"],
            },
            inputs=[],
            outputs=["scoring_data"],
            depends_on=[],
            idempotent=False,  # Data source may change
        )

        # Step 2: load_model
        steps["load_model"] = StepDefinition(
            plugin="mbt.steps.load_model:LoadModelStep",
            config={
                "model_registry": serving_config["model_source"].get("registry", "mlflow"),
                "model_run_id": serving_config["model_source"]["run_id"],
            },
            inputs=[],
            outputs=["model", "model_metadata", "scaler", "encoder", "feature_selector"],
            depends_on=[],
            idempotent=True,  # Same run_id always returns same model
        )

        # Step 3: apply_transforms
        steps["apply_transforms"] = StepDefinition(
            plugin="mbt.steps.apply_transforms:ApplyTransformsStep",
            config={
                "schema": pipeline_config["training"]["schema"],  # Pass schema for column filtering
            },
            inputs=["scoring_data", "scaler", "encoder", "feature_selector"],
            outputs=["transformed_data"],
            depends_on=["load_scoring_data", "load_model"],
            idempotent=True,
        )

        # Step 4: predict
        steps["predict"] = StepDefinition(
            plugin="mbt.steps.predict:PredictStep",
            config={
                "framework": pipeline_config["training"]["model_training"]["framework"],
                "problem_type": pipeline_config["project"]["problem_type"],
            },
            inputs=["model", "transformed_data"],
            outputs=["predictions", "prediction_probabilities"],
            depends_on=["apply_transforms"],
            idempotent=True,
        )

        # Step 5: publish
        output_config = serving_config.get("output", {})
        steps["publish"] = StepDefinition(
            plugin="mbt.steps.publish:PublishStep",
            config={
                "output_config": output_config,
                "run_id": serving_config["model_source"]["run_id"],
            },
            inputs=["predictions", "prediction_probabilities", "scoring_data"],
            outputs=["output_path"],
            depends_on=["predict"],
            idempotent=False,  # Each run writes new predictions
        )

        # Build DAG structure
        dag = self._build_dag_structure(steps)

        return steps, dag

    def _build_dag_structure(self, steps: dict[str, StepDefinition]) -> DAGDefinition:
        """Build DAG structure from step definitions.

        Uses graphlib.TopologicalSorter for ordering and cycle detection.
        """
        # Build parent_map from depends_on
        parent_map = {name: step.depends_on for name, step in steps.items()}

        # Topological sort to get execution order
        sorter = graphlib.TopologicalSorter(parent_map)
        execution_order = list(sorter.static_order())

        # For Phase 1: linear DAG, so each step is its own batch
        # For later: steps in same batch have no dependencies on each other (can run parallel)
        execution_batches = [[step] for step in execution_order]

        return DAGDefinition(
            parent_map=parent_map,
            execution_batches=execution_batches,
        )
