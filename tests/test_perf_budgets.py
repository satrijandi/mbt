"""Performance smoke tests (NFR-03): parse < 2 s and compile < 10 s at 50
resources; per-node execution overhead < 2 s.

These thresholds are deliberately LOOSE - a catastrophic-regression guard (an
accidental O(n^2), a lost cache), not a tight latency budget - so a healthy
build never flakes on them. The measurement is still principled: each is the
BEST of a few runs after a warmup, because a single sample on a shared CI runner
measures the noisy neighbor, not the code (FEEDBACK 2.7). The threshold numbers
are the NFR-03 contract and stay untouched.
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from e2e_utils import DEMO_ANCHOR

from mbt.compile.compiler import CompileOptions, compile_project
from mbt.config.profiles import load_profiles
from mbt.parsing import parse_project

pytestmark = pytest.mark.perf

N_MODELS = 40  # + 10 datasets = 50 resources
MEASURE_RUNS = 3


def best_of(fn: Callable[[], object], runs: int = MEASURE_RUNS) -> float:
    """Minimum wall-clock over `runs` calls, after one untimed warmup."""
    fn()
    best = float("inf")
    for _ in range(runs):
        started = time.monotonic()
        fn()
        best = min(best, time.monotonic() - started)
    return best


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture(scope="module")
def big_project(tmp_path_factory) -> Path:
    project = tmp_path_factory.mktemp("perf") / "big"
    _write(project / "mbt_project.yml", 'name: perf\nversion: "1.0"\n')
    _write(
        project / "profiles.yml",
        # absolute roots: bare `adapter: fake` defaults to ./target/... which
        # resolves against the pytest cwd and litters the repo root
        "perf:\n  target: dev\n  outputs:\n    dev:\n"
        "      data: {adapter: local, config: {root: .}}\n"
        f"      tracking: {{adapter: fake, config: {{root: {project}/target/fake_tracking}}}}\n"
        f"      registry: {{adapter: fake, config: {{root: {project}/target/fake_registry}}}}\n"
        "      compute: {adapter: fake}\n"
        f"      artifact_store: file://{project}/target/artifacts\n",
    )
    _write(
        project / "sources.yml",
        "sources:\n  - name: lake\n    tables:\n"
        "      - name: events\n        path: data/*.parquet\n",
    )
    from datetime import timedelta

    import pyarrow as pa
    import pyarrow.parquet as pq

    # spread across ~180 days so both split windows (vs DEMO_ANCHOR) have rows
    rows = {
        "ts": [datetime(2026, 1, 1) + timedelta(days=i % 180) for i in range(400)],
        "f1": [float(i) for i in range(400)],
        "label": [i % 2 for i in range(400)],
    }
    (project / "data").mkdir()
    pq.write_table(pa.table(rows), project / "data" / "events.parquet")

    for d in range(10):
        _write(
            project / "datasets" / f"ds_{d}.yml",
            f"datasets:\n  - name: ds_{d}\n"
            "    source: source('lake', 'events')\n"
            "    label: {column: label}\n"
            "    split:\n      strategy: temporal\n      time_column: ts\n"
            '      train: "-150d:-28d"\n      test: "-28d:now"\n',
        )
    for m in range(N_MODELS):
        _write(
            project / "models" / f"m_{m}.yml",
            f"models:\n  - name: m_{m}\n"
            "    task: binary_classification\n    adapter: fake\n"
            "    owner: perf@example.com\n"
            f"    dataset: ref('ds_{m % 10}')\n"
            "    target: label\n"
            "    hyperparameters: {max_depth: 3}\n"
            "    evaluation:\n      protocol: {split: temporal}\n"
            "      metrics: [pr_auc]\n"
            f"    seed: {m}\n",
        )
    return project


def test_parse_under_2s_at_50_resources(big_project: Path) -> None:
    assert len(parse_project(big_project).nodes) == 50
    elapsed = best_of(lambda: parse_project(big_project))
    assert elapsed < 2.0, f"parse took {elapsed:.2f}s best-of-{MEASURE_RUNS} (budget 2s, NFR-03)"


def test_compile_under_10s_at_50_resources(big_project: Path) -> None:
    parsed = parse_project(big_project)
    profiles = load_profiles("perf", big_project, project_vars=parsed.project.vars)
    anchor = datetime.fromisoformat(DEMO_ANCHOR.replace("Z", "+00:00")).astimezone(UTC)
    assert len(compile_project(parsed, profiles, options=CompileOptions(anchor=anchor)).nodes) == 50
    elapsed = best_of(
        lambda: compile_project(parsed, profiles, options=CompileOptions(anchor=anchor))
    )
    assert elapsed < 10.0, (
        f"compile took {elapsed:.2f}s best-of-{MEASURE_RUNS} (budget 10s, NFR-03)"
    )


def test_per_node_overhead_under_2s(big_project: Path) -> None:
    """Execution overhead per node, excluding training itself (fake adapter
    training is ~instant, so measured time ≈ overhead). Per-node minimum
    across measured invocations, after one untimed warmup run."""
    from mbt.execute.orchestrator import InvocationOptions, run_command

    anchor = datetime.fromisoformat(DEMO_ANCHOR.replace("Z", "+00:00"))

    def run() -> dict[str, float]:
        results = run_command(
            InvocationOptions(
                command="run",
                project_dir=big_project,
                select=["m_0", "m_1", "m_2"],
                anchor=anchor,
            )
        )
        assert results.exit_code() == 0
        return {r.unique_id: r.execution_time_s for r in results.results}

    run()  # warmup (cold imports, first materialization)
    best: dict[str, float] = {}
    for _ in range(2):
        for uid, elapsed in run().items():
            best[uid] = min(elapsed, best.get(uid, float("inf")))
    for uid, elapsed in best.items():
        assert elapsed < 2.0, f"{uid} overhead {elapsed:.2f}s best-of-2 (budget 2s, NFR-03)"
