"""Selector grammar errors, graph-operator edges, topological order (TSD §9)."""

import networkx as nx
import pytest

from mbt.dag.graph import build_graph, topological_order
from mbt.dag.selector import (
    SelectableNode,
    SelectorError,
    evaluate_selector,
    parse_selector,
)


def _nodes(*names: str) -> dict[str, SelectableNode]:
    return {
        f"model.p.{name}": SelectableNode(
            unique_id=f"model.p.{name}", name=name, resource_type="model"
        )
        for name in names
    }


def _chain(*names: str) -> "nx.DiGraph":
    uids = [f"model.p.{name}" for name in names]
    edges = {uid: uids[i - 1 : i] for i, uid in enumerate(uids)}
    return build_graph(edges, dict.fromkeys(uids, "model"))


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("", "empty selector"),
        ("   ", "empty selector"),
        ("a,", "empty selector atom"),
        ("+", "invalid selector atom"),
        ("owner:me", "unknown selector method 'owner'"),
        ("tag:", "selector method 'tag' needs a value"),
        ("state:stale", "unknown state selector value 'stale'"),
    ],
)
def test_selector_parse_errors(selector: str, message: str) -> None:
    with pytest.raises(SelectorError, match=message):
        parse_selector(selector)


def test_graph_expansion_skips_uids_missing_from_the_graph() -> None:
    nodes = _nodes("a")
    empty_graph = nx.DiGraph()  # 'a' selectable but not a graph node
    assert evaluate_selector("+a", empty_graph, nodes) == {"model.p.a"}
    assert evaluate_selector("a+", empty_graph, nodes) == {"model.p.a"}


def test_depth_limited_descendants() -> None:
    nodes = _nodes("a", "b", "c")
    graph = _chain("a", "b", "c")
    assert evaluate_selector("a+1", graph, nodes) == {"model.p.a", "model.p.b"}
    assert evaluate_selector("a+", graph, nodes) == set(nodes)
    assert evaluate_selector("1+c", graph, nodes) == {"model.p.b", "model.p.c"}


def test_topological_order_without_a_subset_returns_everything() -> None:
    graph = _chain("c", "a", "b")  # dependency order c <- a <- b
    assert topological_order(graph) == ["model.p.c", "model.p.a", "model.p.b"]
    assert topological_order(graph, subset={"model.p.a"}) == ["model.p.a"]
