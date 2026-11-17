"""``mbt registry ls`` (FEEDBACK v7 D-3): what the registry holds, read-only."""

import json
from pathlib import Path

import pytest
from cli_unit_helpers import debug, invoke

from mbt.exceptions import ConfigError
from mbt.registry_view import VersionRow, list_versions
from mbt_adapter_base import ArtifactRef, ModelVersion, Stage
from mbt_adapter_base.capabilities import AdapterKind, DeclaresCapabilities


def _write(demo_project: Path, versions: list[dict]) -> None:
    path = demo_project / "target" / "fake_registry" / "churn_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(versions))


def test_registry_ls_shows_versions_newest_first_with_what_each_recorded(
    demo_project: Path,
) -> None:
    artifact = ArtifactRef(
        uri="file:///m", format="native", content_hash="sha256:" + "ab" * 32, size_bytes=1
    )
    _write(
        demo_project,
        [
            {
                "version": "1",
                "stage": "archived",
                "artifact": artifact.model_dump(mode="json"),
                "tags": {"mbt.gates_passed": "true"},
            },
            {
                "version": "2",
                "stage": "production",
                "artifact": None,
                "tags": {
                    "mbt.gates_passed": "false",
                    "mbt.oot_check.passed": "false",
                    "mbt.oot_check.anchor": "2026-11-01T00:00:00Z",
                },
            },
            {"version": "3", "stage": None, "artifact": None, "tags": {}},
        ],
    )
    result = invoke(
        [
            "registry",
            "ls",
            "--model",
            "churn_model",
            "--project-dir",
            str(demo_project),
            "--output",
            "json",
        ]
    )
    assert result.exit_code == 0, debug(result)
    rows = json.loads(result.stdout)
    assert [r["version"] for r in rows] == ["3", "2", "1"]
    by_version = {r["version"]: r for r in rows}
    assert by_version["3"]["stage"] is None
    assert by_version["3"]["gates"] == "not recorded"
    assert by_version["3"]["after_test"] == "none"
    assert by_version["2"]["stage"] == "production"
    assert by_version["2"]["gates"] == "failed"
    assert by_version["2"]["after_test"] == "failed @ 2026-11-01T00:00:00Z"
    assert by_version["1"]["gates"] == "passed"
    assert by_version["1"]["artifact"] == "ab" * 6


def test_registry_ls_table_and_the_empty_registry(demo_project: Path) -> None:
    empty = invoke(["registry", "ls", "--model", "churn_model", "--project-dir", str(demo_project)])
    assert empty.exit_code == 0, debug(empty)
    assert "no registered versions found" in empty.output

    _write(
        demo_project,
        [
            {
                "version": "1",
                "stage": "staging",
                "artifact": None,
                "tags": {"mbt.gates_passed": "true"},
            }
        ],
    )
    table = invoke(["registry", "ls", "--model", "churn_model", "--project-dir", str(demo_project)])
    assert table.exit_code == 0, debug(table)
    assert "churn_model" in table.output
    assert "staging" in table.output
    assert "passed" in table.output


def test_registry_ls_without_a_model_lists_every_model_the_project_declares(
    demo_project: Path,
) -> None:
    _write(
        demo_project,
        [
            {
                "version": "1",
                "stage": "staging",
                "artifact": None,
                "tags": {"mbt.gates_passed": "true"},
            }
        ],
    )
    result = invoke(["registry", "ls", "--project-dir", str(demo_project), "--output", "json"])
    assert result.exit_code == 0, debug(result)
    assert {r["model"] for r in json.loads(result.stdout)} == {"churn_model"}


def test_a_registry_that_cannot_list_is_refused_with_a_reason() -> None:
    class _NoList:
        def get_version(self, name: str, version: str) -> ModelVersion | None:
            return None

    with pytest.raises(ConfigError, match="cannot list versions") as caught:
        list_versions(_NoList(), ["m"])
    assert "LIST_VERSIONS" in (caught.value.hint or "")


def test_an_after_test_verdict_with_nothing_mature_is_named() -> None:
    class _Listing(DeclaresCapabilities):
        adapter_kind = AdapterKind.REGISTRY

        def list_versions(self, name: str) -> list[ModelVersion]:
            return [
                ModelVersion(
                    name=name,
                    version="1",
                    stage=Stage.STAGING,
                    tags={"mbt.oot_check.passed": "not_gated"},
                )
            ]

    rows = list_versions(_Listing(), ["m"])
    assert rows == [
        VersionRow(
            model="m",
            version="1",
            stage="staging",
            gates="not recorded",
            after_test="nothing mature",
            artifact="",
        )
    ]
