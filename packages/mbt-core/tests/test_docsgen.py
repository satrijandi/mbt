"""Docs generator: model cards carry importance, metrics, gates (FR-DOCS-01..03)."""

import json
from pathlib import Path

import pytest
from core_helpers import TEST_ANCHOR
from test_compile import compile_demo

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import read_manifest
from mbt.artifacts.run_results import RunResults
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
