"""Unit tests for mbt.compile.compiler: resolve errors, snapshots, edge branches."""

from pathlib import Path

import pytest
from core_helpers import TEST_ANCHOR, write
from parse_unit_helpers import ListSink

from mbt.adapters.registry import AdapterRegistry
from mbt.compile.compiler import CompileOptions, build_resolve_context, compile_project
from mbt.config.profiles import load_profiles
from mbt.events import AdapterWarning, EventBus, get_bus, set_bus
from mbt.exceptions import CompilationError
from mbt.parsing import parse_project

DS = "dataset.demo.churn_training"
MODEL = "model.demo.churn_model"


def compile_with_vars(
    project_dir: Path,
    registry: AdapterRegistry,
    parse_vars: dict | None = None,
    compile_vars: dict | None = None,
):
    parsed = parse_project(project_dir, registry=registry, cli_vars=parse_vars)
    profiles = load_profiles(
        "demo", project_dir, cli_vars=parse_vars or {}, project_vars=parsed.project.vars
    )
    return compile_project(
        parsed,
        profiles,
        registry=registry,
        options=CompileOptions(anchor=TEST_ANCHOR),
        cli_vars=compile_vars if compile_vars is not None else parse_vars,
    )


def test_compile_empty_project_with_unknown_data_adapter(
    tmp_path: Path, fake_registry: AdapterRegistry
) -> None:
    """No sources referenced: no snapshot pinning, no adapter versions, live anchor."""
    write(tmp_path / "mbt_project.yml", 'name: tiny\nversion: "1.0"\n')
    write(
        tmp_path / "profiles.yml",
        f"""
        tiny:
          target: dev
          outputs:
            dev:
              data: {{adapter: no_such_adapter_zzz}}
              tracking: {{adapter: fake}}
              registry: {{adapter: fake}}
              artifact_store: file://{tmp_path}/target/artifacts
        """,
    )
    parsed = parse_project(tmp_path, registry=fake_registry)
    profiles = load_profiles("tiny", tmp_path)
    manifest = compile_project(parsed, profiles, registry=fake_registry)  # default options
    assert manifest.nodes == {}
    assert manifest.adapter_versions == {}  # unknown data adapter is skipped
    assert manifest.metadata.generated_at == manifest.metadata.anchor  # live anchor


def test_stale_explicit_snapshot_pin_warns_and_wins(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    dataset = demo_project / "datasets/churn_training.yml"
    dataset.write_text(
        dataset.read_text().replace("checks:", 'snapshot: "sha256:stale"\n    checks:')
    )
    sink = ListSink()
    previous = get_bus()
    set_bus(EventBus([sink]))
    try:
        manifest = compile_with_vars(demo_project, fake_registry)
    finally:
        set_bus(previous)
    assert manifest.nodes[DS].snapshot_id == "sha256:stale"  # the pin wins
    warnings = [e for e in sink.events if isinstance(e, AdapterWarning)]
    assert warnings and "no longer current" in warnings[0].message


def test_resolver_errors_name_the_unknown_resource(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    parsed = parse_project(demo_project, registry=fake_registry)
    profiles = load_profiles("demo", demo_project, project_vars=parsed.project.vars)
    ctx = build_resolve_context(parsed, profiles, {})
    assert ctx.ref_resolver("churn_training") == DS
    assert ctx.source_resolver("lakehouse", "subscribers") == "source.demo.lakehouse.subscribers"
    with pytest.raises(CompilationError, match="ref\\('ghost'\\) does not resolve"):
        ctx.ref_resolver("ghost")
    with pytest.raises(CompilationError, match="source\\('no', 'where'\\) is not declared"):
        ctx.source_resolver("no", "where")


def test_dataset_invalid_after_target_rendering(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    dataset = demo_project / "datasets/churn_training.yml"
    dataset.write_text(
        dataset.read_text().replace("strategy: temporal", "strategy: \"{{ var('strat') }}\"")
    )
    with pytest.raises(CompilationError, match="dataset config invalid after target rendering"):
        compile_with_vars(
            demo_project,
            fake_registry,
            parse_vars={"strat": "temporal"},
            compile_vars={"strat": "sideways"},
        )


def test_model_invalid_after_target_rendering(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model = demo_project / "models/churn_model.yml"
    model.write_text(model.read_text().replace("seed: 42", "seed: \"{{ var('seed_v') }}\""))
    with pytest.raises(CompilationError, match="model config invalid after target rendering"):
        compile_with_vars(
            demo_project,
            fake_registry,
            parse_vars={"seed_v": 42},
            compile_vars={"seed_v": "not_an_int"},
        )


def test_scoring_invalid_after_target_rendering(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "scoring/churn_scoring.yml",
        """
        scoring:
          - name: churn_scoring
            owner: ds@example.com
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            output:
              format: "{{ var('fmt') }}"
              path: predictions/churn
              columns: [user_id]
        """,
    )
    with pytest.raises(CompilationError, match="scoring config invalid after target rendering"):
        compile_with_vars(
            demo_project,
            fake_registry,
            parse_vars={"fmt": "parquet"},
            compile_vars={"fmt": "csv"},
        )


def test_model_test_window_is_resolved(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    model = demo_project / "models/churn_model.yml"
    model.write_text(
        model.read_text().replace(
            "protocol: {split: temporal}",
            'protocol: {split: temporal, test_window: "-14d:now"}',
        )
    )
    manifest = compile_with_vars(demo_project, fake_registry)
    assert manifest.nodes[MODEL].resolved["test_window"] == [
        "2026-06-17T00:00:00Z",
        "2026-07-01T00:00:00Z",
    ]


def test_source_format_the_adapter_cannot_read_is_a_compilation_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    """A referenced 'delta' source must fail compile on a parquet-only adapter (F23)."""
    sources = demo_project / "sources.yml"
    sources.write_text(
        sources.read_text().replace(
            "path: data/subscribers/*.parquet",
            "path: data/subscribers/*.parquet\n        format: delta",
        )
    )
    with pytest.raises(CompilationError, match="cannot read") as excinfo:
        compile_with_vars(demo_project, fake_registry)
    assert "lakehouse.subscribers" in excinfo.value.message
    assert "'delta'" in excinfo.value.message
    assert "spark" in (excinfo.value.hint or "")


def test_source_format_the_adapter_supports_compiles(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch
) -> None:
    """An adapter declaring delta support accepts a delta source (F23)."""

    class _DeltaCapableAdapter:
        supported_source_formats = frozenset({"parquet", "delta"})

        def snapshot_id(self, table, deep=False):
            return "sha256:delta-snap"

    monkeypatch.setattr(
        "mbt.compile.compiler.build_data_adapter",
        lambda profiles, project_dir, registry: _DeltaCapableAdapter(),
    )
    sources = demo_project / "sources.yml"
    sources.write_text(
        sources.read_text().replace(
            "path: data/subscribers/*.parquet",
            "path: data/subscribers/*.parquet\n        format: delta",
        )
    )
    manifest = compile_with_vars(demo_project, fake_registry)
    assert manifest.nodes[DS].snapshot_id == "sha256:delta-snap"


def test_snapshot_pinning_failure_is_a_compilation_error(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch
) -> None:
    class _RaisingAdapter:
        def snapshot_id(self, table, deep=False):
            raise RuntimeError("disk on fire")

    def fake_build(profiles, project_dir, registry):
        return _RaisingAdapter()

    monkeypatch.setattr("mbt.compile.compiler.build_data_adapter", fake_build)
    with pytest.raises(CompilationError, match="snapshot pinning failed") as excinfo:
        compile_with_vars(demo_project, fake_registry)
    assert "lakehouse.subscribers" in excinfo.value.message
    assert "disk on fire" in excinfo.value.message


def test_scoring_inputs_snapshot_combines_spine_and_features(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: extra_features
                path: data/subscribers/*.parquet
        """,
    )
    write(
        demo_project / "scoring/inputs_form.yml",
        """
        scoring:
          - name: sc_inputs
            owner: ds@example.com
            model: ref('churn_model')
            input:
              inputs:
                spine: source('lakehouse', 'subscribers')
                features: ["source('lakehouse', 'extra_features')"]
                join_key: user_id
            output: {path: predictions/scores, columns: [user_id]}
        """,
    )
    manifest = compile_with_vars(demo_project, fake_registry)
    node = manifest.nodes["scoring.demo.sc_inputs"]
    assert node.snapshot_id and node.snapshot_id.startswith("sha256:")
    assert node.snapshot_id != manifest.nodes[DS].snapshot_id  # combined, not single
