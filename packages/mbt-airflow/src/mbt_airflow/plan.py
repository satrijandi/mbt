"""Manifest + selector -> task specs. Pure: it imports mbt, never Airflow (ADR-33).

This is the part of the package that decides what the DAG IS. Everything in
``operators`` and ``dag`` only carries that decision out, so the decision can be
read, tested, and reused without an Airflow installation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from mbt.artifacts.manifest import Manifest
from mbt.exceptions import ConfigError
from mbt.execute.planner import plan_execution

#: Task id of the root task every other task hangs off.
COMPILE = "mbt_compile"

#: The node kinds each command executes. This is mbt's own rule (the planner's
#: ``executable=``), not a choice made here: ``build`` never sees scoring nodes
#: and ``score``/``monitor`` never see datasets or models. Drawing both kinds in
#: one DAG would put a daily score behind a weekly retrain (ADR-33).
UNIVERSE: dict[str, tuple[str, ...]] = {
    "run": ("dataset", "model"),
    "build": ("dataset", "model"),
    "test": ("dataset", "model"),
    "score": ("scoring",),
    "monitor": ("scoring",),
}

#: Commands whose node tasks read and write the shared dataset cache.
DATASET_COMMANDS = frozenset({"run", "build", "test"})


@dataclass(frozen=True)
class TaskSpec:
    """One task of the DAG, before it is an Airflow object."""

    task_id: str
    #: The node's unique id; empty for the compile task.
    uid: str
    #: ``compile``, ``dataset``, ``model`` or ``scoring``.
    kind: str
    #: The mbt command the task runs (``compile``, ``build``, ``score``, ...).
    command: str
    #: Task ids this one waits for.
    upstream: tuple[str, ...]


def plan_tasks(
    manifest: Manifest,
    command: str,
    select: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[TaskSpec]:
    """The tasks of a DAG that runs ``command`` over ``select`` of ``manifest``.

    The compile task comes first and is the parent of every root. Nodes follow in
    mbt's own topological order, so the result is deterministic.

    ``select`` is the CLI's selector grammar, evaluated by mbt's planner, so a
    model selection pulls in its upstream datasets exactly as ``mbt build`` would
    (ADR-13). That closure is not optional here: every node task runs with
    ``--no-auto-materialize`` (ADR-32), so a model whose dataset is not a task of
    its own would find an empty cache. Edges are the manifest's ``depends_on``
    restricted to the plan; for ``score`` that drops scoring -> model, because a
    scoring node reads the registry's champion at run time (ADR-20) and does not
    wait for a training run.

    ``state:`` selectors are refused by the planner: a DAG is drawn before any
    run, so there is no reference state to compare against.
    """
    if command not in UNIVERSE:
        raise ConfigError(
            f"cannot render the {command!r} command as tasks",
            hint=f"supported commands: {', '.join(sorted(UNIVERSE))}",
        )
    plan = plan_execution(
        manifest,
        list(select) if select else None,
        list(exclude) if exclude else None,
        executable=UNIVERSE[command],
    )
    if not plan.order:
        raise ConfigError(
            f"the selection {list(select or [])} matched no {'/'.join(UNIVERSE[command])} "
            f"nodes for 'mbt {command}'",
            hint="a DAG with only a compile task would run nothing behind a green run; "
            "select nodes the command executes, or use the other command",
        )

    task_ids = {uid: manifest.nodes[uid].name for uid in plan.order}
    names = sorted(task_ids.values())
    if len(set(names)) != len(names):
        raise ConfigError(
            f"two selected nodes share a name, so their task ids would collide: {names}",
            hint="rename one resource, or select fewer nodes",
        )

    graph = manifest.graph()
    specs = [TaskSpec(COMPILE, "", "compile", "compile", ())]
    for uid in plan.order:
        parents = sorted(
            task_ids[parent] for parent in graph.predecessors(uid) if parent in task_ids
        )
        specs.append(
            TaskSpec(
                task_id=task_ids[uid],
                uid=uid,
                kind=manifest.nodes[uid].resource_type,
                command=command,
                upstream=tuple(parents) or (COMPILE,),
            )
        )
    return specs


__all__ = ["COMPILE", "DATASET_COMMANDS", "UNIVERSE", "TaskSpec", "plan_tasks"]
