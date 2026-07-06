"""Property-based selector algebra tests over random DAGs (S4-03, TSD §21)."""

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

from mbt.dag.selector import SelectableNode, evaluate_selector, select_nodes

_TAGS = ("alpha", "beta", "weekly")


@st.composite
def random_dag(draw):
    """A small random DAG of datasets and models with tags."""
    n = draw(st.integers(min_value=2, max_value=12))
    graph = nx.DiGraph()
    nodes: dict[str, SelectableNode] = {}
    for i in range(n):
        kind = draw(st.sampled_from(["dataset", "model"]))
        uid = f"{kind}.p.n{i}"
        tags = tuple(t for t in _TAGS if draw(st.booleans()))
        graph.add_node(uid)
        nodes[uid] = SelectableNode(unique_id=uid, name=f"n{i}", resource_type=kind, tags=tags)
        # edges only from lower to higher index: guaranteed acyclic
        uids = sorted(nodes)
        for previous in uids[:-1]:
            if draw(st.integers(min_value=0, max_value=3)) == 0:
                graph.add_edge(previous, uid)
    return graph, nodes


@given(random_dag(), st.sampled_from(_TAGS))
@settings(max_examples=60, deadline=None)
def test_intersection_is_subset_of_each_atom(dag, tag) -> None:
    """a,b ⊆ a and a,b ⊆ b."""
    graph, nodes = dag
    a = evaluate_selector(f"tag:{tag}", graph, nodes)
    b = evaluate_selector("resource_type:model", graph, nodes)
    both = evaluate_selector(f"tag:{tag},resource_type:model", graph, nodes)
    assert both <= a
    assert both <= b
    assert both == a & b


@given(random_dag(), st.sampled_from(_TAGS))
@settings(max_examples=60, deadline=None)
def test_union_is_superset_of_each_atom(dag, tag) -> None:
    graph, nodes = dag
    a = evaluate_selector(f"tag:{tag}", graph, nodes)
    b = evaluate_selector("resource_type:dataset", graph, nodes)
    union = evaluate_selector(f"tag:{tag} resource_type:dataset", graph, nodes)
    assert union == a | b


@given(random_dag())
@settings(max_examples=60, deadline=None)
def test_graph_operators_are_monotone(dag) -> None:
    """base ⊆ +base ⊆ +base+ and depth-limited ⊆ unlimited."""
    graph, nodes = dag
    base = evaluate_selector("n0", graph, nodes)
    up = evaluate_selector("+n0", graph, nodes)
    both = evaluate_selector("+n0+", graph, nodes)
    up1 = evaluate_selector("1+n0", graph, nodes)
    assert base <= up <= both
    assert base <= up1 <= up


@given(random_dag())
@settings(max_examples=60, deadline=None)
def test_exclude_subtracts_exactly(dag) -> None:
    graph, nodes = dag
    everything = select_nodes(graph, nodes, None)
    models = evaluate_selector("resource_type:model", graph, nodes)
    remaining = select_nodes(graph, nodes, None, exclude=["resource_type:model"])
    assert remaining == everything - models


@given(random_dag())
@settings(max_examples=60, deadline=None)
def test_star_glob_selects_everything(dag) -> None:
    graph, nodes = dag
    assert evaluate_selector("*", graph, nodes) == set(nodes)
