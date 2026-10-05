"""Selector grammar errors, graph-operator edges, topological order (TSD §9)."""

import networkx as nx
import pytest

from mbt.dag.graph import build_graph, topological_order
from mbt.dag.selector import (
    SelectableNode,
    SelectorError,
    evaluate_selector,
    parse_selector,
    select_nodes,
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


def _tagged() -> dict[str, SelectableNode]:
    return {
        "model.p.churn_classifier": SelectableNode(
            unique_id="model.p.churn_classifier",
            name="churn_classifier",
            resource_type="model",
            tags=("weekly",),
        ),
        "scoring.p.churn_scoring": SelectableNode(
            unique_id="scoring.p.churn_scoring",
            name="churn_scoring",
            resource_type="scoring",
            tags=("daily",),
        ),
    }


@pytest.mark.parametrize(
    ("flag", "selector", "message"),
    [
        ("select", "churn_clasifier", "--select 'churn_clasifier' matches no resource"),
        ("select", "tag:nightly", "--select 'tag:nightly' matches no tag"),
        ("select", "resource_type:modle", "matches no resource_type"),
        # one bad atom in a union fails the whole selection
        ("select", "tag:daily tag:nightly", "'tag:nightly' matches no tag"),
        # a mistyped exclude would run the very thing it was written to skip
        ("exclude", "tag:weeky", "--exclude 'tag:weeky' matches no tag"),
    ],
)
def test_a_selector_naming_nothing_is_an_error(flag: str, selector: str, message: str) -> None:
    """FEEDBACK v6 A-5: renaming a tag made scheduled scoring select 0 nodes,
    exit 0, and ping its heartbeat while scoring nothing."""
    with pytest.raises(SelectorError, match=message):
        select_nodes(nx.DiGraph(), _tagged(), **{"select": None, flag: [selector]})


def test_the_error_suggests_the_closest_name() -> None:
    with pytest.raises(SelectorError) as info:
        select_nodes(nx.DiGraph(), _tagged(), ["tag:weeky"])
    assert info.value.hint and info.value.hint.startswith("did you mean 'weekly'? known tags:")


def test_a_legitimately_empty_selection_is_not_an_error() -> None:
    # each atom names something; they just share nothing
    assert select_nodes(nx.DiGraph(), _tagged(), ["tag:weekly,tag:daily"]) == set()


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
