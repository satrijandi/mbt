"""Docs generator: model cards carry importance, metrics, gates (FR-DOCS-01..03)."""

import json
from pathlib import Path

import pytest
from core_helpers import TEST_ANCHOR
from test_compile import compile_demo

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import read_manifest
from mbt.artifacts.run_results import (
    RunResults,
    command_results_path,
    read_latest_results,
)
from mbt.docsgen.generator import generate_docs
from mbt.execute.orchestrator import InvocationOptions, run_command


def test_model_card_renders_feature_importance(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    run_command(
        InvocationOptions(command="run", project_dir=demo_project, anchor=TEST_ANCHOR),
        registry=fake_registry,
    )
    manifest = read_manifest(demo_project / "target" / "manifest.json", source="test")
    results = RunResults.model_validate(
        json.loads((demo_project / "target" / "run_results.json").read_text())
    )
    index = generate_docs(manifest, results, demo_project / "target" / "docs")
    assert index.name == "index.html"

    card = (demo_project / "target" / "docs" / "model_churn_model.html").read_text()
    assert "Feature importance" in card
    assert "fake_signal" in card and "75.0%" in card
    assert "Metrics (latest run)" in card


def test_partial_dependence_renders_as_a_sparkline() -> None:
    """The card shows a per-feature partial-dependence sparkline and a low->high
    summary (explainability); no PD data yields no section."""
    from mbt.artifacts.run_results import NodeResult
    from mbt.docsgen.generator import _partial_dependence_section

    result = NodeResult(
        unique_id="model.p.m",
        status="success",
        partial_dependence={"monthly_usage": [[0.0, 0.6], [50.0, 0.4], [100.0, 0.2]]},
    )
    section = _partial_dependence_section(result)
    assert "Partial dependence" in section and "monthly_usage" in section
    assert "<polyline" in section and "svg" in section  # the sparkline
    assert "0.600" in section and "0.200" in section  # low -> high summary

    empty = NodeResult(unique_id="model.p.m", status="success")
    assert _partial_dependence_section(empty) == ""


def test_metric_table_shows_walk_forward_backtest_beside_single_split() -> None:
    """The card juxtaposes each metric's single-split value with its walk-forward
    backtest mean +/- fold std (R2-7), so an optimistic single test window and an
    unstable-across-folds estimate are both obvious."""
    from mbt.artifacts.run_results import NodeResult
    from mbt.docsgen.generator import _metric_table

    result = NodeResult(
        unique_id="model.p.m",
        status="success",
        metrics={"pr_auc": 0.81, "logloss": 0.30},
        backtest_metrics={"pr_auc": 0.72},  # logloss deliberately absent
        backtest_std={"pr_auc": 0.05},
    )
    table = _metric_table(result)
    assert "backtest (cross-validated mean &pm; std)" in table
    assert "0.8100" in table and "0.7200" in table  # single-split vs backtest, side by side
    assert "<td>0.7200 &pm; 0.0500</td>" in table  # the mean carries its fold-to-fold std
    assert ">-<" in table  # logloss has no backtest value -> a dash cell

    # A backtest mean without a std (older results) still renders, bare (the
    # header still names std, but the cell carries no `&pm;`).
    bare = _metric_table(
        NodeResult(
            unique_id="model.p.m",
            status="success",
            metrics={"pr_auc": 0.81},
            backtest_metrics={"pr_auc": 0.72},
        )
    )
    assert "<td>0.7200</td>" in bare

    # No backtest -> the extra column is absent (unchanged for existing cards).
    plain = _metric_table(
        NodeResult(unique_id="model.p.m", status="success", metrics={"pr_auc": 0.8})
    )
    assert "backtest" not in plain


def test_model_card_shows_auto_keyword_not_sentinel(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A hyperparameter declared ``auto`` renders as ``auto`` in the published
    card, not the internal ``__mbt_auto__`` token the manifest keeps verbatim."""
    model = demo_project / "models" / "churn_model.yml"
    model.write_text(
        model.read_text().replace(
            "max_depth: 4", 'max_depth: 4\n      scale_pos_weight: "{{ auto }}"'
        )
    )
    run_command(
        InvocationOptions(command="run", project_dir=demo_project, anchor=TEST_ANCHOR),
        registry=fake_registry,
    )
    manifest = read_manifest(demo_project / "target" / "manifest.json", source="test")
    generate_docs(manifest, None, demo_project / "target" / "docs")

    card = (demo_project / "target" / "docs" / "model_churn_model.html").read_text()
    assert "__mbt_auto__" not in card, "internal AUTO sentinel leaked into the published card"
    assert "<code>auto</code>" in card  # the keyword the user wrote


def test_docs_generate_redacts_env_var_secret(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mbt docs generate` (no --manifest) compiles fresh, so generate_docs
    sees the raw tainted config; a spec env_var() value must not reach the
    published HTML (docs deploy to GitHub Pages)."""
    secret = "sk-DOCS-LEAK-7777"
    monkeypatch.setenv("MBT_DOCS_SECRET", secret)
    model = demo_project / "models" / "churn_model.yml"
    model.write_text(
        model.read_text().replace(
            "owner: ds@example.com", "owner: \"{{ env_var('MBT_DOCS_SECRET') }}\""
        )
    )
    # compile_project (what the default docs path uses) yields an in-memory
    # manifest whose config is NOT yet redacted - only the file write redacts.
    manifest = compile_demo(demo_project, fake_registry)
    generate_docs(manifest, None, demo_project / "target" / "docs")

    card = (demo_project / "target" / "docs" / "model_churn_model.html").read_text()
    index = (demo_project / "target" / "docs" / "index.html").read_text()
    assert secret not in card, "secret leaked into the published model card"
    assert secret not in index, "secret leaked into the published index"
    assert "***" in card and "***" in index  # the owner field was masked


def test_model_card_keeps_metrics_after_a_scoring_run(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """The A-2 regression, driven through the real orchestrator.

    `mbt score`/`mbt monitor` rewrite the shared run_results.json with only
    their own nodes, so a docs run afterwards used to render "no run results
    yet - run mbt build" at a user who had just built. The card must survive
    any command running in between.
    """
    run_command(
        InvocationOptions(command="run", project_dir=demo_project, anchor=TEST_ANCHOR),
        registry=fake_registry,
    )
    results_path = demo_project / "target" / "run_results.json"
    build_results = json.loads(results_path.read_text())
    assert any(r["metrics"] for r in build_results["results"])

    # Stand in for a later serving command: same shared file, no model nodes.
    shared = json.loads(results_path.read_text())
    shared["metadata"]["command"] = "monitor"
    shared["results"] = []
    results_path.write_text(json.dumps(shared))
    (demo_project / "target" / "run_results.monitor.json").write_text(json.dumps(shared))

    manifest = read_manifest(demo_project / "target" / "manifest.json", source="test")
    recovered = read_latest_results(results_path, commands=("build", "run", "evaluate"))
    assert recovered is not None
    generate_docs(manifest, recovered, demo_project / "target" / "docs")

    card = (demo_project / "target" / "docs" / "model_churn_model.html").read_text()
    assert "no metrics for this model" not in card
    assert "pr_auc" in card


def test_read_latest_results_falls_back_to_the_shared_file(demo_project: Path) -> None:
    """A project that has never run, and one whose only file is the shared one."""
    results_path = demo_project / "target" / "run_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    assert read_latest_results(results_path, commands=("build",)) is None

    metadata = {
        "run_id": "r1",
        "mbt_version": "0.1.0",
        "target": "dev",
        "manifest_hash": "sha256:x",
        "anchor": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:00Z",
        "command": "build",
    }
    results_path.write_text(json.dumps({"metadata": metadata, "results": []}))
    loaded = read_latest_results(results_path, commands=("build",))
    assert loaded is not None and loaded.metadata.command == "build"
    # no `commands` at all is the "whatever ran last" reading
    assert read_latest_results(results_path) is not None


def test_write_emits_both_the_shared_file_and_the_command_sibling(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    run_command(
        InvocationOptions(command="run", project_dir=demo_project, anchor=TEST_ANCHOR),
        registry=fake_registry,
    )
    target = demo_project / "target"
    assert (target / "run_results.json").is_file()
    assert (target / "run_results.run.json").is_file()
    assert (target / "run_results.json").read_text() == (
        target / "run_results.run.json"
    ).read_text()
    assert command_results_path(target / "run_results.json", "score").name == (
        "run_results.score.json"
    )
