"""DAG scheduler: parallel branches, skip propagation, fail-fast (TSD §10.2).

Threads only coordinate; training runs in subprocesses (ADR-3), so the GIL
never serializes real work. A node starts when all its in-plan parents
succeeded; a failure marks all transitive downstream ``skipped`` while
independent branches continue; ``--fail-fast`` cancels pending work.
"""

import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import networkx as nx

from mbt.artifacts.run_results import NodeResult
from mbt.execute.planner import ExecutionPlan

_FAILING_STATUSES = frozenset({"error", "gate_failed", "test_failed"})


def execute_plan(
    plan: ExecutionPlan,
    graph: "nx.DiGraph",
    run_node: Callable[[str], NodeResult],
    *,
    threads: int = 1,
    fail_fast: bool = False,
) -> dict[str, NodeResult]:
    """Run every node in the plan; returns a result for each one."""
    execution_set = set(plan.execution_set)
    results: dict[str, NodeResult] = {}
    lock = threading.Lock()
    stop_scheduling = False

    parents = {
        uid: {p for p in graph.predecessors(uid) if p in execution_set} for uid in execution_set
    }
    remaining = {uid: len(deps) for uid, deps in parents.items()}
    children = {
        uid: {c for c in graph.successors(uid) if c in execution_set} for uid in execution_set
    }

    def guarded_run(uid: str) -> NodeResult:
        try:
            return run_node(uid)
        except Exception as exc:
            return NodeResult(unique_id=uid, status="error", message=str(exc))

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures: dict[Future[NodeResult], str] = {}

        def submit_ready() -> None:
            nonlocal stop_scheduling
            running = {futures[f] for f in futures}
            for uid in sorted(execution_set):
                if stop_scheduling or len(futures) >= max(1, threads):
                    # Keep at most `threads` outstanding so --fail-fast can
                    # actually cancel work that has not been handed out yet.
                    break
                if uid in results or remaining[uid] > 0 or uid in running:
                    continue
                futures[pool.submit(guarded_run, uid)] = uid
                running.add(uid)

        def mark_skipped(uid: str, reason: str) -> None:
            if uid not in results:
                results[uid] = NodeResult(unique_id=uid, status="skipped", message=reason)

        submit_ready()
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            with lock:
                for future in done:
                    uid = futures.pop(future)
                    result = future.result()
                    results[uid] = result
                    if result.status in _FAILING_STATUSES:
                        for descendant in nx.descendants(graph, uid):
                            if descendant in execution_set:
                                mark_skipped(descendant, f"upstream {uid} {result.status}")
                        if fail_fast:
                            stop_scheduling = True
                    for child in children[uid]:
                        remaining[child] -= 1
                submit_ready()

        if stop_scheduling:
            for uid in execution_set:
                mark_skipped(uid, "cancelled by --fail-fast")

    # Anything never scheduled (e.g. parents skipped) resolves to skipped.
    for uid in execution_set:
        mark_skipped(uid, "upstream never completed")
    return results
