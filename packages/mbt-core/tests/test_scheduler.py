"""Scheduler semantics with in-process runners: parallelism, skip, fail-fast (S5-05)."""

import threading
import time

import networkx as nx

from mbt.artifacts.run_results import NodeResult
from mbt.execute.planner import ExecutionPlan
from mbt.execute.scheduler import execute_plan


def _plan(nodes: set[str], order: list[str]) -> ExecutionPlan:
    return ExecutionPlan(
        selected=frozenset(nodes), execution_set=frozenset(nodes), order=tuple(order)
    )


def _graph(edges: list[tuple[str, str]], nodes: set[str]) -> "nx.DiGraph":
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def test_topological_execution_and_results() -> None:
    executed: list[str] = []
    graph = _graph([("a", "b"), ("b", "c")], {"a", "b", "c"})

    def run(uid: str) -> NodeResult:
        executed.append(uid)
        return NodeResult(unique_id=uid, status="success")

    results = execute_plan(_plan({"a", "b", "c"}, ["a", "b", "c"]), graph, run)
    assert executed == ["a", "b", "c"]
    assert all(r.status == "success" for r in results.values())


def test_failure_skips_downstream_but_not_independent_branches() -> None:
    graph = _graph(
        [("ds", "m1"), ("m1", "m2"), ("ds2", "m3")], {"ds", "m1", "m2", "ds2", "m3"}
    )

    def run(uid: str) -> NodeResult:
        if uid == "m1":
            return NodeResult(unique_id=uid, status="error", message="boom")
        return NodeResult(unique_id=uid, status="success")

    results = execute_plan(
        _plan({"ds", "m1", "m2", "ds2", "m3"}, ["ds", "ds2", "m1", "m3", "m2"]), graph, run
    )
    assert results["m1"].status == "error"
    assert results["m2"].status == "skipped"
    assert "m1" in (results["m2"].message or "")
    assert results["m3"].status == "success"  # independent branch continues


def test_gate_failure_also_skips_downstream() -> None:
    graph = _graph([("a", "b")], {"a", "b"})

    def run(uid: str) -> NodeResult:
        return NodeResult(unique_id=uid, status="gate_failed" if uid == "a" else "success")

    results = execute_plan(_plan({"a", "b"}, ["a", "b"]), graph, run)
    assert results["b"].status == "skipped"


def test_fail_fast_cancels_pending_work() -> None:
    graph = _graph([], {"a", "b", "c", "d"})
    started: list[str] = []
    gate = threading.Event()

    def run(uid: str) -> NodeResult:
        started.append(uid)
        if uid == "a":
            return NodeResult(unique_id=uid, status="error")
        gate.wait(0.05)
        return NodeResult(unique_id=uid, status="success")

    results = execute_plan(
        _plan({"a", "b", "c", "d"}, ["a", "b", "c", "d"]), graph, run, threads=1,
        fail_fast=True,
    )
    assert results["a"].status == "error"
    skipped = [uid for uid, r in results.items() if r.status == "skipped"]
    assert skipped, "fail-fast should cancel pending nodes"


def test_parallel_branches_actually_overlap() -> None:
    graph = _graph([], {"a", "b"})
    concurrent = {"count": 0, "max": 0}
    lock = threading.Lock()

    def run(uid: str) -> NodeResult:
        with lock:
            concurrent["count"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["count"])
        time.sleep(0.05)
        with lock:
            concurrent["count"] -= 1
        return NodeResult(unique_id=uid, status="success")

    execute_plan(_plan({"a", "b"}, ["a", "b"]), graph, run, threads=2)
    assert concurrent["max"] == 2


def test_crashing_runner_becomes_error_result() -> None:
    graph = _graph([], {"a"})

    def run(uid: str) -> NodeResult:
        raise RuntimeError("runner crashed")

    results = execute_plan(_plan({"a"}, ["a"]), graph, run)
    assert results["a"].status == "error"
    assert "runner crashed" in (results["a"].message or "")
