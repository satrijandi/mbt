"""``MbtDag`` and ``MbtTaskGroup``: an mbt project drawn as Airflow tasks (ADR-33).

The DAG is drawn at parse time from a manifest compiled ahead of time (CI would
publish it beside the project); the tasks compile again at run time at the run's
own anchor, and every node task executes that manifest verbatim.
"""

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from airflow.sdk import DAG, Param, TaskGroup

from mbt.artifacts.manifest import read_manifest
from mbt.exceptions import ConfigError
from mbt_airflow.operators import (
    MbtCleanupOperator,
    MbtCompileOperator,
    MbtNodeOperator,
)
from mbt_airflow.plan import plan_tasks
from mbt_airflow.runners import MbtDocker

#: Environment variable naming the default workspace root.
WORKSPACE_ENV = "MBT_AIRFLOW_WORKSPACE"

#: A run's anchor when nothing pins one: the end of the data interval, else the
#: logical date, else when the run was created. Manual runs have no interval.
_DERIVED_ANCHOR = "(data_interval_end or logical_date or run_after).isoformat()"


@dataclass(frozen=True)
class MbtProject:
    """Which mbt project to draw, and which profile to run it against."""

    #: The project root on the worker (where ``mbt_project.yml`` is).
    project_dir: str | os.PathLike[str]
    #: A compiled ``manifest.json`` the graph is drawn from. Only its shape is
    #: used at parse time; each run compiles its own.
    manifest_path: str | os.PathLike[str]
    #: ``--target``; None means the profile's own default.
    target: str | None = None
    #: ``--profiles-dir``; None means mbt's own search order.
    profiles_dir: str | os.PathLike[str] | None = None


@dataclass(frozen=True)
class MbtExecution:
    """How the tasks run: ``mbt`` on the worker, or in a container (``docker``)."""

    mbt_executable: str = "mbt"
    #: Where each run's private target directories and shared dataset cache live.
    #: Every worker must see it, so on more than one machine it is a shared volume.
    #: Defaults to ``$MBT_AIRFLOW_WORKSPACE``, else a temp directory; with ``docker``
    #: it is a path in the container and has no default, because the worker's temp
    #: directory is not a place a container can see.
    workspace_root: str | os.PathLike[str] | None = None
    #: Extra environment for the mbt processes. On the worker the worker's own is
    #: inherited; in a container nothing is, and ``DockerUnit.env`` is the base.
    env: Mapping[str, str] | None = None
    #: Run every mbt process in a container instead of on the worker.
    docker: MbtDocker | None = None
    #: A hard error (exit 1) is retried this many times; a quality verdict (exit 2)
    #: never is.
    retries: int = 1
    retry_delay: timedelta = timedelta(seconds=30)
    #: Add a final task that removes the run's workspace, but only if every node
    #: succeeded, so a failed run keeps what you need to debug it.
    cleanup_on_success: bool = False
    #: Passed to every operator (``pool``, ``queue``, ``execution_timeout``, ...).
    operator_args: Mapping[str, Any] = field(default_factory=dict)


def _workspace_root(execution: MbtExecution) -> str:
    if execution.workspace_root is not None:
        return os.fspath(execution.workspace_root)
    if execution.docker is not None and not os.environ.get(WORKSPACE_ENV):
        raise ConfigError(
            "MbtExecution.workspace_root is required with docker",
            hint="name the directory inside the container where the run keeps its state, "
            "a path on the unit's shared volume (for example /workspace/mbt-airflow)",
        )
    return os.environ.get(WORKSPACE_ENV) or str(Path(tempfile.gettempdir()) / "mbt-airflow")


def _workspace(root: str, scope: str) -> str:
    """The run's workspace as a template: one directory per DAG run (and group)."""
    run = "{{ run_id | replace(':', '-') | replace('+', '_') }}"
    parts = [root, "{{ dag.dag_id }}", run, *([scope] if scope else [])]
    return "/".join(parts)


def _dag_anchor(anchor_command: Sequence[str] | None) -> str:
    """The anchor template of a DAG: the param, else the command, else the run's own."""
    if anchor_command:
        return "{{ params.anchor }}"  # empty renders empty, and the compile task asks the command
    return f"{{{{ params.anchor or {_DERIVED_ANCHOR} }}}}"


def _add_tasks(
    *,
    project: MbtProject,
    execution: MbtExecution,
    command: str,
    select: Sequence[str] | None,
    exclude: Sequence[str] | None,
    anchor: str,
    anchor_command: Sequence[str] | None,
    mbt_vars: str,
    scope: str,
) -> str:
    """Create the tasks inside the current DAG or group; return the project's name."""
    manifest = read_manifest(Path(project.manifest_path), source="the manifest to draw")
    specs = plan_tasks(manifest, command, select, exclude)
    root = _workspace_root(execution)
    workspace = _workspace(root, scope)
    common: dict[str, Any] = {
        "project_dir": os.fspath(project.project_dir),
        "workspace": workspace,
        "anchor": anchor,
        "mbt_vars": mbt_vars,
        "mbt_executable": execution.mbt_executable,
        "target": project.target,
        "profiles_dir": os.fspath(project.profiles_dir) if project.profiles_dir else None,
        "env": execution.env,
        "docker": execution.docker,
        "retries": execution.retries,
        "retry_delay": execution.retry_delay,
        **execution.operator_args,
    }
    rendered = [spec.task_id for spec in specs]
    tasks: dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "compile":
            tasks[spec.task_id] = MbtCompileOperator(
                task_id=spec.task_id,
                plan_command=command,
                plan_select=select,
                plan_exclude=exclude,
                rendered_for=rendered,
                anchor_command=anchor_command,
                **common,
            )
        else:
            tasks[spec.task_id] = MbtNodeOperator(task_id=spec.task_id, spec=spec, **common)
    for spec in specs:
        for upstream in spec.upstream:
            tasks[upstream] >> tasks[spec.task_id]

    if execution.cleanup_on_success:
        cleanup = MbtCleanupOperator(
            task_id="mbt_cleanup",
            workspace=workspace,
            workspace_root=root,
            docker=execution.docker,
        )
        waited_on = {up for spec in specs for up in spec.upstream}
        for spec in specs:
            if spec.task_id not in waited_on:
                tasks[spec.task_id] >> cleanup
    return manifest.metadata.project_name


class MbtDag(DAG):
    """An mbt project as a DAG: one task per node, edges from the manifest.

    ``command`` is what runs: ``build`` (also ``run``, ``test``) draws datasets and
    models, ``score`` and ``monitor`` draw scoring nodes. They are separate DAGs
    because mbt's own commands partition the nodes that way (ADR-33).

    ``select`` and ``exclude`` use the CLI's selector grammar. The run's anchor is
    the ``anchor`` DAG param when set, else ``anchor`` here, else what
    ``anchor_command`` prints, else derived from the run; ``vars`` is the other
    param, forwarded to the compile task only.

    ``anchor_command`` is for a project whose clock is its data: a command run
    where mbt runs (in the container, with ``MbtDocker``) that prints the anchor.
    """

    def __init__(
        self,
        dag_id: str,
        *,
        project: MbtProject,
        execution: MbtExecution | None = None,
        command: str = "build",
        select: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        anchor: str | None = None,
        anchor_command: Sequence[str] | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        **dag_kwargs: Any,
    ) -> None:
        dag_kwargs.setdefault("schedule", None)
        dag_kwargs.setdefault("catchup", False)
        super().__init__(
            dag_id=dag_id,
            params={
                **(params or {}),
                "anchor": Param(anchor or "", type="string"),
                "vars": Param("", type="string"),
            },
            tags=list(tags or []),
            **dag_kwargs,
        )
        with self:
            project_name = _add_tasks(
                project=project,
                execution=execution or MbtExecution(),
                command=command,
                select=select,
                exclude=exclude,
                anchor=_dag_anchor(anchor_command),
                anchor_command=anchor_command,
                mbt_vars="{{ params.vars }}",
                scope="",
            )
        for tag in ("mbt", command, project_name):
            self.tags.add(tag)


class MbtTaskGroup(TaskGroup):
    """The same tasks inside a DAG you already have, to chain with your own.

    ``anchor`` and ``mbt_vars`` are literals or Jinja templates; the anchor
    defaults to what ``anchor_command`` prints, else one derived from the run. Task
    ids are prefixed by the group, and the group's workspace is its own, so several
    groups can share a DAG.
    """

    def __init__(
        self,
        group_id: str,
        *,
        project: MbtProject,
        execution: MbtExecution | None = None,
        command: str = "build",
        select: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        anchor: str | None = None,
        anchor_command: Sequence[str] | None = None,
        mbt_vars: str = "",
        **group_kwargs: Any,
    ) -> None:
        super().__init__(group_id=group_id, **group_kwargs)
        with self:
            _add_tasks(
                project=project,
                execution=execution or MbtExecution(),
                command=command,
                select=select,
                exclude=exclude,
                anchor=anchor or ("" if anchor_command else f"{{{{ {_DERIVED_ANCHOR} }}}}"),
                anchor_command=anchor_command,
                mbt_vars=mbt_vars,
                scope=group_id,
            )


__all__ = ["WORKSPACE_ENV", "MbtDag", "MbtExecution", "MbtProject", "MbtTaskGroup"]
