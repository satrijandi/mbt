"""Orchestrator plugin contract for workflow schedulers.

Orchestrator plugins generate native DAG files for workflow schedulers
(Airflow, Prefect, Dagster, etc.) from MBT manifest files.
"""

from abc import ABC, abstractmethod


class OrchestratorPlugin(ABC):
    """Abstract base class for orchestrator adapters.

    Orchestrator plugins convert MBT manifests into scheduler-native DAG files,
    enabling MBT pipelines to run on production workflow schedulers.

    Example:
        >>> orchestrator = plugin_registry.get("mbt.orchestrators", "airflow")
        >>> orchestrator.generate_dag_file(
        ...     manifest_path="target/my_pipeline/manifest.json",
        ...     output_path="dags/ml_my_pipeline.py",
        ...     schedule="@daily"
        ... )
    """

    @abstractmethod
    def generate_dag_file(
        self,
        manifest_path: str,
        output_path: str,
        schedule: str | None = None,
        **kwargs
    ) -> None:
        """Generate orchestrator-native DAG file from MBT manifest.

        Args:
            manifest_path: Path to compiled manifest.json
            output_path: Path to write DAG file
            schedule: Schedule expression (cron or preset)
                     Airflow: "@daily", "@hourly", "0 0 * * *"
                     Prefect: "0 0 * * *"
            **kwargs: Additional orchestrator-specific parameters

        Example:
            For Airflow:
            >>> orchestrator.generate_dag_file(
            ...     manifest_path="target/churn_training_v1/manifest.json",
            ...     output_path="dags/ml_churn_training.py",
            ...     schedule="@daily",
            ...     owner="ml_team",
            ...     retries=2
            ... )

            Generates Airflow Python file:
            ```python
            from airflow.sdk import DAG
            from airflow.providers.standard.operators.bash import BashOperator

            dag = DAG(dag_id="ml_churn_training_v1", schedule="@daily", ...)

            load_data = BashOperator(
                task_id="load_data",
                bash_command="mbt exec --pipeline churn_training_v1 --step load_data",
                dag=dag
            )
            # ... more tasks
            load_data >> split_data >> train_model
            ```
        """
        pass

    def validate_manifest(self, manifest_path: str) -> bool:
        """Validate that manifest is compatible with this orchestrator.

        Args:
            manifest_path: Path to manifest.json

        Returns:
            True if compatible, False otherwise

        Note:
            Default implementation always returns True. Override for
            orchestrator-specific validation (resource limits, etc.).
        """
        return True

    def get_schedule_presets(self) -> dict[str, str]:
        """Get common schedule presets for this orchestrator.

        Returns:
            Dictionary mapping preset names to cron expressions

        Note:
            Default implementation returns common presets. Override for
            orchestrator-specific schedules.

        Example:
            >>> orchestrator.get_schedule_presets()
            {
                "@hourly": "0 * * * *",
                "@daily": "0 0 * * *",
                "@weekly": "0 0 * * 0",
                "@monthly": "0 0 1 * *"
            }
        """
        return {
            "@hourly": "0 * * * *",
            "@daily": "0 0 * * *",
            "@weekly": "0 0 * * 0",
            "@monthly": "0 0 1 * *",
        }
