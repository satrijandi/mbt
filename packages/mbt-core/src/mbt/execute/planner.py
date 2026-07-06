"""Execution planning (TSD §10.1, FR-RUN-12, ADR-13).

Selection governs which models *train*; every dataset a selected model needs
joins the execution plan even if unselected (datasets are cheap
materializations and CI runners start cold).
"""

from dataclasses import dataclass

import networkx as nx

from mbt.artifacts.manifest import Manifest
from mbt.dag.graph import topological_order
from mbt.dag.selector import StateIndex, select_nodes

_EXECUTABLE = ("dataset", "model")


@dataclass(frozen=True)
class ExecutionPlan:
    selected: frozenset[str]  # what the user asked to build/train
    execution_set: frozenset[str]  # selected + required upstream datasets
    order: tuple[str, ...]  # deterministic topological order

    @property
    def auto_materialized(self) -> frozenset[str]:
        return self.execution_set - self.selected


def plan_execution(
    manifest: Manifest,
    select: list[str] | None,
    exclude: list[str] | None,
    state: StateIndex | None = None,
) -> ExecutionPlan:
    graph = manifest.graph()
    selectable = manifest.selectable_nodes()
    selected = select_nodes(graph, selectable, select, exclude, state)
    selected = {
        uid
        for uid in selected
        if manifest.nodes.get(uid) is not None and manifest.nodes[uid].resource_type in _EXECUTABLE
    }

    execution_set = set(selected)
    for uid in selected:
        node = manifest.nodes[uid]
        if node.resource_type != "model":
            continue
        for ancestor in nx.ancestors(graph, uid):
            upstream = manifest.nodes.get(ancestor)
            if upstream is not None and upstream.resource_type == "dataset":
                execution_set.add(ancestor)

    order = topological_order(graph, subset=execution_set)
    return ExecutionPlan(
        selected=frozenset(selected),
        execution_set=frozenset(execution_set),
        order=tuple(order),
    )
