"""Unit tests for gate-verified promotion (mbt/promote.py) and the
``mbt promote`` command surface (registry seeded on disk, no training)."""

import json
from pathlib import Path

import pytest
from cli_unit_helpers import (  # noqa: F401 - autouse fixture import
    cli_process_state,
    debug,
    install_recording_bus,
    invoke,
)

from mbt.contracts import ModelVersion, Stage
from mbt.events.models import LogMessage, PromotionApplied
from mbt.exceptions import ConfigError, StateError
from mbt.promote import PromotionEntry, load_promotions_file, promote_model

# -- helpers ------------------------------------------------------------------------


def version(name: str = "m", number: str = "1", *, gates: str = "true") -> ModelVersion:
    return ModelVersion(
        name=name,
        version=number,
        stage=Stage.STAGING,
        tags={"mbt.gates_passed": gates},
    )


class StubRegistry:
    def __init__(
        self,
        versions: dict[tuple[str, str], ModelVersion] | None = None,
        champion: ModelVersion | None = None,
    ) -> None:
        self.versions = versions or {}
        self.champion = champion
        self.transitions: list[tuple[ModelVersion, Stage]] = []
        self.champion_queries: list[tuple[str, Stage]] = []

    def get_version(self, name: str, number: str) -> ModelVersion | None:
        return self.versions.get((name, number))

    def get_champion(self, name: str, stage: Stage) -> ModelVersion | None:
        self.champion_queries.append((name, stage))
        return self.champion

    def transition(self, resolved: ModelVersion, stage: Stage) -> None:
        self.transitions.append((resolved, stage))


# -- promote_model ------------------------------------------------------------------


def test_promote_explicit_version_with_recorded_gates() -> None:
    sink = install_recording_bus()
    resolved = version(number="2")
    registry = StubRegistry(versions={("m", "2"): resolved})
    outcome = promote_model(registry, name="m", to_stage=Stage.PRODUCTION, version="2")
    assert outcome.name == "m"
    assert outcome.version == "2"
    assert outcome.to_stage is Stage.PRODUCTION
    assert outcome.forced is False
    assert registry.transitions == [(resolved, Stage.PRODUCTION)]
    applied = [event for event in sink.events if isinstance(event, PromotionApplied)]
    assert len(applied) == 1
    assert applied[0].forced is False


def test_promote_unknown_version_is_a_state_error() -> None:
    registry = StubRegistry()
    with pytest.raises(StateError, match="no version '9'"):
        promote_model(registry, name="m", to_stage=Stage.PRODUCTION, version="9")
    assert registry.transitions == []


def test_promote_defaults_to_the_staging_champion() -> None:
    registry = StubRegistry(champion=version())
    outcome = promote_model(registry, name="m", to_stage=Stage.PRODUCTION)
    assert outcome.version == "1"
    assert registry.champion_queries == [("m", Stage.STAGING)]


def test_promote_without_any_staged_version_is_a_state_error() -> None:
    registry = StubRegistry(champion=None)
    with pytest.raises(StateError, match="no version in stage 'staging'"):
        promote_model(registry, name="m", to_stage=Stage.PRODUCTION)


def test_promote_refuses_when_gates_did_not_pass() -> None:
    registry = StubRegistry(champion=version(gates="false"))
    with pytest.raises(StateError, match="gates were not"):
        promote_model(registry, name="m", to_stage=Stage.PRODUCTION)
    assert registry.transitions == []


def test_forced_promotion_warns_and_marks_outcome() -> None:
    sink = install_recording_bus()
    resolved = version(gates="false")
    registry = StubRegistry(champion=resolved)
    outcome = promote_model(registry, name="m", to_stage=Stage.PRODUCTION, force=True)
    assert outcome.forced is True
    assert registry.transitions == [(resolved, Stage.PRODUCTION)]
    warnings = [
        event for event in sink.events if isinstance(event, LogMessage) and event.level == "warn"
    ]
    assert warnings and "FORCED promotion" in warnings[0].message
    applied = [event for event in sink.events if isinstance(event, PromotionApplied)]
    assert applied[0].forced is True


# -- load_promotions_file -----------------------------------------------------------


def test_load_promotions_file_parses_entries(tmp_path: Path) -> None:
    path = tmp_path / "promotions.yml"
    path.write_text(
        "promotions:\n"
        "  - model: churn_model\n"
        "    to: production\n"
        "    version: '3'\n"
        "  - model: upsell_model\n"
        "    to: archived\n"
    )
    entries = load_promotions_file(path)
    assert entries == [
        PromotionEntry(model="churn_model", to=Stage.PRODUCTION, version="3"),
        PromotionEntry(model="upsell_model", to=Stage.ARCHIVED, version=None),
    ]


def test_load_promotions_file_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_promotions_file(tmp_path / "promotions.yml")


def test_load_promotions_file_rejects_bad_yaml(tmp_path: Path) -> None:
    path = tmp_path / "promotions.yml"
    path.write_text("promotions: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid promotions file"):
        load_promotions_file(path)


def test_load_promotions_file_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "promotions.yml"
    path.write_text("promotions: [{model: m, to: production, bogus: 1}]\n")
    with pytest.raises(ConfigError, match="invalid promotions file"):
        load_promotions_file(path)


# -- mbt promote (CLI) --------------------------------------------------------------


def seed_registry(demo_project: Path, *, gates: str = "true") -> Path:
    """One staging version of churn_model in the fake registry's on-disk shape."""
    registry_dir = demo_project / "target" / "fake_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / "churn_model.json"
    path.write_text(
        json.dumps(
            [
                {
                    "version": "1",
                    "stage": "staging",
                    "artifact": None,
                    "tags": {"mbt.gates_passed": gates},
                }
            ]
        )
    )
    return path


def test_promote_cli_transitions_a_version(demo_project: Path) -> None:
    registry_file = seed_registry(demo_project)
    result = invoke(
        [
            "promote",
            "--model",
            "churn_model",
            "--to",
            "production",
            "--project-dir",
            str(demo_project),
        ]
    )
    assert result.exit_code == 0, debug(result)
    assert "promoted" in result.output
    assert json.loads(registry_file.read_text())[0]["stage"] == "production"


def test_promote_cli_requires_model_and_to(demo_project: Path) -> None:
    result = invoke(["promote", "--project-dir", str(demo_project)])
    assert result.exit_code == 1, debug(result)
    assert "needs --model" in result.stderr


def test_promote_cli_rejects_unknown_stage(demo_project: Path) -> None:
    result = invoke(
        [
            "promote",
            "--model",
            "churn_model",
            "--to",
            "nowhere",
            "--project-dir",
            str(demo_project),
        ]
    )
    assert result.exit_code == 1, debug(result)
    assert "unknown stage" in result.stderr


def test_promote_cli_applies_a_reviewed_promotions_file(demo_project: Path) -> None:
    registry_file = seed_registry(demo_project)
    promotions = demo_project / "promotions.yml"
    promotions.write_text(
        "promotions:\n  - model: churn_model\n    to: production\n    version: '1'\n"
    )
    result = invoke(["promote", "--from-file", str(promotions), "--project-dir", str(demo_project)])
    assert result.exit_code == 0, debug(result)
    assert "applied 1 promotion(s)" in result.output
    assert json.loads(registry_file.read_text())[0]["stage"] == "production"
