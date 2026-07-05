"""Graph building, cycle detection, topological order (TSD §9.1)."""

import networkx as nx

from mbt.exceptions import ConfigError


def build_graph(edges: dict[str, list[str]], node_types: dict[str, str]) -> "nx.DiGraph":
    """Build the DAG: edge ``u -> v`` means "v depends on u".

    ``edges`` maps each node's unique_id to the unique_ids it depends on.
    """
    graph = nx.DiGraph()
    for uid, resource_type in node_types.items():
        graph.add_node(uid, resource_type=resource_type)
    for uid, depends_on in edges.items():
        for dep in depends_on:
            graph.add_edge(dep, uid)
    return graph


def find_cycle(graph: "nx.DiGraph") -> list[str] | None:
    """The full path of one dependency cycle, if any (FR-DAG-01)."""
    try:
        cycle_edges = nx.find_cycle(graph, orientation="original")
    except nx.NetworkXNoCycle:
        return None
    path = [edge[0] for edge in cycle_edges]
    path.append(cycle_edges[-1][1])
    return path


def ensure_acyclic(graph: "nx.DiGraph") -> None:
    cycle = find_cycle(graph)
    if cycle is not None:
        raise ConfigError(
            "dependency cycle detected: " + " -> ".join(cycle),
            hint="break the cycle by removing one of the ref() edges above",
        )


def topological_order(graph: "nx.DiGraph", subset: set[str] | None = None) -> list[str]:
    """Deterministic topological order (lexicographic tie-break)."""
    order = list(nx.lexicographical_topological_sort(graph))
    if subset is None:
        return order
    return [uid for uid in order if uid in subset]
