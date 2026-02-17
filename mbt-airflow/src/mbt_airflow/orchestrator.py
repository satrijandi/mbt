"""Airflow orchestrator adapter for MBT.

Generates Airflow DAG Python files from MBT manifest files.
Each pipeline step becomes an Airflow task using BashOperator, DockerOperator,
or KubernetesPodOperator depending on executor type.
"""

import json
from pathlib import Path

from mbt.contracts.orchestrator import OrchestratorPlugin


# Map deployment cadence to Airflow schedule
CADENCE_MAP = {
    "hourly": "@hourly",
    "daily": "@daily",
    "weekly": "@weekly",
    "monthly": "@monthly",
}


class AirflowOrchestrator(OrchestratorPlugin):
    """Generates Airflow DAG files from MBT manifests.

    For local/docker execution: generates BashOperator tasks.
    For Kubernetes execution: generates KubernetesPodOperator tasks.
    """

    def generate_dag_file(
        self,
        manifest_path: str,
        output_path: str,
        schedule: str | None = None,
        **kwargs,
    ) -> None:
        """Generate an Airflow DAG Python file from manifest.

        Args:
            manifest_path: Path to compiled manifest.json
            output_path: Path to write the DAG Python file
            schedule: Schedule expression (e.g., "@daily", "@monthly")
            **kwargs: Additional config:
                - owner: DAG owner (default: "mbt")
                - retries: Number of retries (default: 2)
                - retry_delay_minutes: Retry delay in minutes (default: 5)
                - executor_type: "local" or "kubernetes" (default: "local")
                - image: Docker image for KubernetesPodOperator
                - namespace: K8s namespace
                - service_account: K8s service account
        """
        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        pipeline_name = manifest["metadata"]["pipeline_name"]
        target = manifest["metadata"]["target"]
        dag_definition = manifest["dag"]
        parent_map = dag_definition["parent_map"]
        profile_config = manifest.get("profile_config", {})

        # Determine schedule
        if schedule is None:
            # Try to get from deployment config
            deployment = manifest.get("metadata", {})
            cadence = kwargs.get("cadence", "daily")
            schedule = CADENCE_MAP.get(cadence, "@daily")

        # Get orchestrator config
        owner = kwargs.get("owner", "mbt")
        retries = kwargs.get("retries", 2)
        retry_delay_minutes = kwargs.get("retry_delay_minutes", 5)

        # Determine executor type
        executor_config = profile_config.get("executor", {})
        executor_type = kwargs.get("executor_type", executor_config.get("type", "local"))

        # Generate DAG content
        exec_cfg = executor_config.get("config", {})

        if executor_type == "kubernetes":
            dag_content = self._generate_k8s_dag(
                pipeline_name=pipeline_name,
                target=target,
                parent_map=parent_map,
                schedule=schedule,
                owner=owner,
                retries=retries,
                retry_delay_minutes=retry_delay_minutes,
                image=kwargs.get("image", exec_cfg.get("image", "mbt-runner:latest")),
                namespace=kwargs.get("namespace", exec_cfg.get("namespace", "mbt-pipelines")),
                service_account=kwargs.get("service_account", exec_cfg.get("service_account", "mbt-runner")),
            )
        elif executor_type == "docker":
            dag_content = self._generate_docker_dag(
                pipeline_name=pipeline_name,
                target=target,
                parent_map=parent_map,
                schedule=schedule,
                owner=owner,
                retries=retries,
                retry_delay_minutes=retry_delay_minutes,
                image=kwargs.get("image", exec_cfg.get("image", "mbt-runner:latest")),
                network_mode=kwargs.get("network_mode", exec_cfg.get("network_mode", "bridge")),
                project_mount_source=kwargs.get("project_host_dir") or exec_cfg.get("project_mount_source"),
                project_mount_target=kwargs.get("project_dir") or exec_cfg.get("project_mount_target", "/opt/mbt/project"),
                environment=exec_cfg.get("environment", {}),
                docker_url=kwargs.get("docker_url", exec_cfg.get("docker_url", "unix://var/run/docker.sock")),
            )
        else:
            # Filter kwargs to only pass extra keys not already explicit
            extra_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ("owner", "retries", "retry_delay_minutes",
                                        "executor_type", "cadence")}
            dag_content = self._generate_bash_dag(
                pipeline_name=pipeline_name,
                target=target,
                parent_map=parent_map,
                schedule=schedule,
                owner=owner,
                retries=retries,
                retry_delay_minutes=retry_delay_minutes,
                **extra_kwargs,
            )

        # Write DAG file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dag_content)

    def _generate_bash_dag(
        self,
        pipeline_name: str,
        target: str,
        parent_map: dict,
        schedule: str,
        owner: str,
        retries: int,
        retry_delay_minutes: int,
        **kwargs,
    ) -> str:
        """Generate DAG with BashOperator tasks."""
        steps = list(parent_map.keys())

        # Get project directory for working directory
        project_dir = kwargs.get("project_dir")

        # Build task definitions
        task_defs = []
        for step_name in steps:
            if project_dir:
                bash_cmd = (
                    f'            "cd {project_dir} &&"\n'
                    f'            " mbt step execute"\n'
                )
            else:
                bash_cmd = f'            "mbt step execute"\n'
            task_defs.append(
                f'    {step_name} = BashOperator(\n'
                f'        task_id="{step_name}",\n'
                f'        bash_command=(\n'
                + bash_cmd +
                f'            " --pipeline {pipeline_name}"\n'
                f'            " --step {step_name}"\n'
                f'            " --target {target}"\n'
                f'            \' --run-id "run_{{{{ ds_nodash }}}}_{{{{ ts_nodash }}}}"\'\n'
                f'        ),\n'
                f'    )\n'
            )

        # Build dependency lines
        dep_lines = []
        for step_name, parents in parent_map.items():
            for parent in parents:
                dep_lines.append(f"    {parent} >> {step_name}")

        return (
            f'"""\n'
            f'Auto-generated by MBT. Do not edit.\n'
            f'Pipeline: {pipeline_name}\n'
            f'Target: {target}\n'
            f'"""\n'
            f'\n'
            f'from airflow import DAG\n'
            f'from airflow.operators.bash import BashOperator\n'
            f'from datetime import datetime, timedelta\n'
            f'\n'
            f'default_args = {{\n'
            f'    "owner": "{owner}",\n'
            f'    "retries": {retries},\n'
            f'    "retry_delay": timedelta(minutes={retry_delay_minutes}),\n'
            f'}}\n'
            f'\n'
            f'with DAG(\n'
            f'    dag_id="{pipeline_name}",\n'
            f'    default_args=default_args,\n'
            f'    schedule="{schedule}",\n'
            f'    start_date=datetime(2026, 1, 1),\n'
            f'    catchup=False,\n'
            f'    tags=["mbt", "{pipeline_name}"],\n'
            f') as dag:\n'
            f'\n'
            + "\n".join(task_defs)
            + "\n"
            + "    # Dependencies\n"
            + ("\n".join(dep_lines) if dep_lines else "    pass")
            + "\n"
        )

    def _generate_docker_dag(
        self,
        pipeline_name: str,
        target: str,
        parent_map: dict,
        schedule: str,
        owner: str,
        retries: int,
        retry_delay_minutes: int,
        image: str,
        network_mode: str,
        project_mount_source: str | None,
        project_mount_target: str,
        environment: dict,
        docker_url: str,
    ) -> str:
        """Generate DAG with DockerOperator tasks."""
        steps = list(parent_map.keys())

        # Build environment dict string
        env_str = ""
        if environment:
            env_items = ",\n".join(
                f'            "{k}": "{v}"' for k, v in environment.items()
            )
            env_str = (
                f"        environment={{\n"
                f"{env_items},\n"
                f"        }},\n"
            )

        # Build mount definition (outside the DAG context)
        mount_def = ""
        mounts_arg = ""
        if project_mount_source:
            mount_def = (
                f'_project_mount = Mount(\n'
                f'    target="{project_mount_target}",\n'
                f'    source="{project_mount_source}",\n'
                f'    type="bind",\n'
                f'    read_only=False,\n'
                f')\n\n'
            )
            mounts_arg = f"        mounts=[_project_mount],\n"

        # Build task definitions
        task_defs = []
        for step_name in steps:
            task_defs.append(
                f'    {step_name} = DockerOperator(\n'
                f'        task_id="{step_name}",\n'
                f'        image="{image}",\n'
                f'        command=(\n'
                f'            "step execute"\n'
                f'            " --pipeline {pipeline_name}"\n'
                f'            " --step {step_name}"\n'
                f'            " --target {target}"\n'
                f'            \' --run-id "run_{{{{ ds_nodash }}}}_{{{{ ts_nodash }}}}"\'\n'
                f'        ),\n'
                f'        working_dir="{project_mount_target}",\n'
                + mounts_arg
                + f'        network_mode="{network_mode}",\n'
                + env_str
                + f'        docker_url="{docker_url}",\n'
                f'        auto_remove=True,\n'
                f'        mount_tmp_dir=False,\n'
                f'    )\n'
            )

        # Build dependency lines
        dep_lines = []
        for step_name, parents in parent_map.items():
            for parent in parents:
                dep_lines.append(f"    {parent} >> {step_name}")

        # Build imports
        imports = (
            f'from airflow import DAG\n'
            f'from airflow.providers.docker.operators.docker import DockerOperator\n'
        )
        if project_mount_source:
            imports += f'from docker.types import Mount\n'
        imports += f'from datetime import datetime, timedelta\n'

        return (
            f'"""\n'
            f'Auto-generated by MBT. Do not edit.\n'
            f'Pipeline: {pipeline_name}\n'
            f'Target: {target}\n'
            f'"""\n'
            f'\n'
            + imports
            + f'\n'
            f'default_args = {{\n'
            f'    "owner": "{owner}",\n'
            f'    "retries": {retries},\n'
            f'    "retry_delay": timedelta(minutes={retry_delay_minutes}),\n'
            f'}}\n'
            f'\n'
            + mount_def
            + f'with DAG(\n'
            f'    dag_id="{pipeline_name}",\n'
            f'    default_args=default_args,\n'
            f'    schedule="{schedule}",\n'
            f'    start_date=datetime(2026, 1, 1),\n'
            f'    catchup=False,\n'
            f'    tags=["mbt", "{pipeline_name}"],\n'
            f') as dag:\n'
            f'\n'
            + "\n".join(task_defs)
            + "\n"
            + "    # Dependencies\n"
            + ("\n".join(dep_lines) if dep_lines else "    pass")
            + "\n"
        )

    def _generate_k8s_dag(
        self,
        pipeline_name: str,
        target: str,
        parent_map: dict,
        schedule: str,
        owner: str,
        retries: int,
        retry_delay_minutes: int,
        image: str,
        namespace: str,
        service_account: str,
    ) -> str:
        """Generate DAG with KubernetesPodOperator tasks."""
        steps = list(parent_map.keys())

        # Build task definitions
        task_defs = []
        for step_name in steps:
            task_defs.append(
                f'    {step_name} = KubernetesPodOperator(\n'
                f'        task_id="{step_name}",\n'
                f'        name="mbt-{pipeline_name}-{step_name}",\n'
                f'        namespace="{namespace}",\n'
                f'        image="{image}",\n'
                f'        cmds=["mbt"],\n'
                f'        arguments=[\n'
                f'            "step", "execute",\n'
                f'            "--pipeline", "{pipeline_name}",\n'
                f'            "--step", "{step_name}",\n'
                f'            "--target", "{target}",\n'
                f'            "--run-id", "run_{{{{ ds_nodash }}}}_{{{{ ts_nodash }}}}",\n'
                f'        ],\n'
                f'        service_account_name="{service_account}",\n'
                f'        is_delete_operator_pod=True,\n'
                f'        get_logs=True,\n'
                f'    )\n'
            )

        # Build dependency lines
        dep_lines = []
        for step_name, parents in parent_map.items():
            for parent in parents:
                dep_lines.append(f"    {parent} >> {step_name}")

        return (
            f'"""\n'
            f'Auto-generated by MBT. Do not edit.\n'
            f'Pipeline: {pipeline_name}\n'
            f'Target: {target}\n'
            f'"""\n'
            f'\n'
            f'from airflow import DAG\n'
            f'from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator\n'
            f'from datetime import datetime, timedelta\n'
            f'\n'
            f'default_args = {{\n'
            f'    "owner": "{owner}",\n'
            f'    "retries": {retries},\n'
            f'    "retry_delay": timedelta(minutes={retry_delay_minutes}),\n'
            f'}}\n'
            f'\n'
            f'with DAG(\n'
            f'    dag_id="{pipeline_name}",\n'
            f'    default_args=default_args,\n'
            f'    schedule="{schedule}",\n'
            f'    start_date=datetime(2026, 1, 1),\n'
            f'    catchup=False,\n'
            f'    tags=["mbt", "{pipeline_name}"],\n'
            f') as dag:\n'
            f'\n'
            + "\n".join(task_defs)
            + "\n"
            + "    # Dependencies\n"
            + ("\n".join(dep_lines) if dep_lines else "    pass")
            + "\n"
        )
