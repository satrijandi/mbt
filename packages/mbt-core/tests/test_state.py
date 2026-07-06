"""state:modified / state:new selection and mbt state diff (S6-01, S6-02)."""

from datetime import timedelta
from pathlib import Path

import pytest
from core_helpers import TEST_ANCHOR, write, write_subscriber_data
from test_compile import DS, MODEL, compile_demo

from mbt.adapters.registry import AdapterRegistry
from mbt.dag.selector import SelectorError, select_nodes
from mbt.state.diff import ManifestStateIndex, diff_manifests


def _select_modified(current, reference, selector: str = "state:modified") -> set[str]:
    index = ManifestStateIndex(current, reference)
    return select_nodes(current.graph(), current.selectable_nodes(), [selector], state=index)


def test_anchor_drift_selects_nothing(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    reference = compile_demo(demo_project, fake_registry)
    drifted = compile_demo(demo_project, fake_registry, anchor=TEST_ANCHOR + timedelta(days=14))
    assert _select_modified(drifted, reference) == set()
    diff = diff_manifests(drifted, reference)
    assert diff.is_empty and not diff.env_changed


def test_spec_edit_selects_model_and_reports_config_component(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reference = compile_demo(demo_project, fake_registry)
    model_yml = demo_project / "models/churn_model.yml"
    model_yml.write_text(model_yml.read_text().replace("max_depth: 4", "max_depth: 6"))
    current = compile_demo(demo_project, fake_registry)

    assert _select_modified(current, reference) == {MODEL}
    diff = diff_manifests(current, reference)
    assert [d.unique_id for d in diff.modified] == [MODEL]
    assert diff.modified[0].components == ("config",)


def test_snapshot_change_selects_dataset_and_model(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reference = compile_demo(demo_project, fake_registry)
    write_subscriber_data(demo_project, n_rows=450)
    current = compile_demo(demo_project, fake_registry)

    assert _select_modified(current, reference) == {DS, MODEL}
    diff = diff_manifests(current, reference)
    components = {d.unique_id: d.components for d in diff.modified}
    assert "snapshot" in components[DS]
    assert components[MODEL] == ("upstream",)


def test_hooks_edit_selects_model_with_hooks_component(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "models/churn_model.py",
        """
        def transform_features(table, ctx):
            return table
        """,
    )
    reference = compile_demo(demo_project, fake_registry)
    write(
        demo_project / "models/churn_model.py",
        """
        def transform_features(table, ctx):
            return table.drop_columns(["monthly_usage"])
        """,
    )
    current = compile_demo(demo_project, fake_registry)
    assert _select_modified(current, reference) == {MODEL}
    diff = diff_manifests(current, reference)
    assert diff.modified[0].components == ("hooks",)


def test_state_new_selects_only_added_nodes(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reference = compile_demo(demo_project, fake_registry)
    write(
        demo_project / "models/second_model.yml",
        """
        models:
          - name: second_model
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc]
            seed: 9
        """,
    )
    current = compile_demo(demo_project, fake_registry)
    index = ManifestStateIndex(current, reference)
    new = select_nodes(current.graph(), current.selectable_nodes(), ["state:new"], state=index)
    assert new == {"model.demo.second_model"}
    diff = diff_manifests(current, reference)
    assert [d.unique_id for d in diff.added] == ["model.demo.second_model"]


def test_state_selector_without_state_flag_is_an_error(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    manifest = compile_demo(demo_project, fake_registry)
    with pytest.raises(SelectorError, match="--state"):
        select_nodes(manifest.graph(), manifest.selectable_nodes(), ["state:modified"], state=None)


def test_state_modified_plus_includes_downstream(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    reference = compile_demo(demo_project, fake_registry)
    ds_yml = demo_project / "datasets/churn_training.yml"
    ds_yml.write_text(ds_yml.read_text().replace('"is_active = true"', '"tenure_days >= 30"'))
    current = compile_demo(demo_project, fake_registry)
    selected = _select_modified(current, reference, "state:modified+")
    assert selected == {DS, MODEL}
    # transitive input_hash means plain state:modified matches both already (ADR-4)
    assert _select_modified(current, reference) == {DS, MODEL}
