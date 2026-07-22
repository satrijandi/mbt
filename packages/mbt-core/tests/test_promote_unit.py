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

from mbt.contracts import ArtifactRef, ModelVersion, Stage
from mbt.events.models import LogMessage, PromotionApplied
from mbt.exceptions import ConfigError, StateError
from mbt.promote import PromotionEntry, load_promotions_file, promote_model, rollback_model

# -- helpers ------------------------------------------------------------------------

#: A rollback target needs an artifact reference, or F12 refuses. The scheme is
#: deliberately unprobeable so the head probe reports "cannot verify" and the
#: rollback proceeds with a warning (a file:// path would have to really exist).
_ARTIFACT = ArtifactRef(
    uri="memory://m/model.ubj", format="xgboost_ubj", content_hash="sha256:a", size_bytes=1
)


def version(
    name: str = "m",
    number: str = "1",
    *,
    gates: str = "true",
    artifact: ArtifactRef | None = _ARTIFACT,
) -> ModelVersion:
    return ModelVersion(
        name=name,
        version=number,
        stage=Stage.STAGING,
        artifact=artifact,
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


# -- rollback_model -----------------------------------------------------------------


def _rollback_registry(champion_number: str, prior: dict[str, str]) -> StubRegistry:
    """A stub whose production champion is ``champion_number`` and whose earlier
    versions are ``prior`` (number -> gates-passed string)."""
    versions = {("m", n): version(number=n, gates=g) for n, g in prior.items()}
    return StubRegistry(versions=versions, champion=version(number=champion_number))


def test_rollback_auto_selects_last_gated_version_below_champion() -> None:
    sink = install_recording_bus()
    reg = _rollback_registry("3", {"2": "true", "1": "true"})
    outcome = rollback_model(reg, name="m")
    assert outcome.version == "2"  # highest gated version below v3
    assert reg.champion_queries == [("m", Stage.PRODUCTION)]
    assert reg.transitions[-1][0].version == "2" and reg.transitions[-1][1] is Stage.PRODUCTION
    warns = [e for e in sink.events if isinstance(e, LogMessage) and "ROLLBACK" in e.message]
    assert warns and "reverted from v3 to v2" in warns[0].message


def test_rollback_skips_ungated_versions() -> None:
    # v3 is the champion, v2 never passed gates, v1 did -> land on v1.
    reg = _rollback_registry("3", {"2": "false", "1": "true"})
    assert rollback_model(reg, name="m").version == "1"


def test_rollback_explicit_version() -> None:
    reg = _rollback_registry("3", {"2": "true", "1": "true"})
    assert rollback_model(reg, name="m", to_version="1").version == "1"


def test_rollback_without_production_champion_is_a_state_error() -> None:
    reg = StubRegistry(champion=None)
    with pytest.raises(StateError, match="no production champion"):
        rollback_model(reg, name="m")
    assert reg.transitions == []


def test_rollback_with_no_earlier_gated_version_is_a_state_error() -> None:
    reg = _rollback_registry("2", {"1": "false"})  # only prior never passed gates
    with pytest.raises(StateError, match="no earlier gated version"):
        rollback_model(reg, name="m")
    assert reg.transitions == []


def test_rollback_to_the_current_champion_is_refused() -> None:
    reg = _rollback_registry("3", {"3": "true", "2": "true"})
    with pytest.raises(StateError, match="already the production champion"):
        rollback_model(reg, name="m", to_version="3")


def test_rollback_refuses_a_target_with_no_loadable_artifact() -> None:
    # v2 is the champion; v1 passed its gates but its artifact was aged out
    # (artifact=None, e.g. by `mbt clean`). Rollback must refuse rather than move
    # the alias and leave the next `mbt score` to die with 'no loadable
    # artifact' (F12).
    reg = StubRegistry(
        versions={("m", "1"): version(number="1", artifact=None)},
        champion=version(number="2"),
    )
    with pytest.raises(StateError, match="no loadable artifact"):
        rollback_model(reg, name="m", to_version="1")
    assert reg.transitions == []  # the alias never moved


def test_rollback_refuses_a_target_whose_artifact_file_is_gone(tmp_path: Path) -> None:
    # The REF surviving in the registry proves nothing: the file behind it can
    # be gone (`mbt clean`, a bucket lifecycle rule). F12's head probe refuses
    # at the rollback command, not at the next `mbt score`.
    dangling = ArtifactRef(
        uri=f"file://{tmp_path}/gone.ubj",
        format="xgboost_ubj",
        content_hash="sha256:a",
        size_bytes=1,
    )
    reg = StubRegistry(
        versions={("m", "1"): version(number="1", artifact=dangling)},
        champion=version(number="2"),
    )
    with pytest.raises(StateError, match="no longer exists"):
        rollback_model(reg, name="m", to_version="1")
    assert reg.transitions == []  # the alias never moved

    # write the file and the same rollback goes through
    (tmp_path / "gone.ubj").write_bytes(b"m")
    outcome = rollback_model(reg, name="m", to_version="1")
    assert outcome.version == "1" and reg.transitions


def test_rollback_warns_but_proceeds_on_an_unprobeable_artifact_scheme() -> None:
    # An unrecognized scheme (or s3 without the extra) cannot be head-probed;
    # blocking an incident on that would be worse than proceeding loudly.
    reg = StubRegistry(
        versions={("m", "1"): version(number="1")},  # memory:// default fixture
        champion=version(number="2"),
    )
    sink = install_recording_bus()
    outcome = rollback_model(reg, name="m", to_version="1")
    assert outcome.version == "1"
    warns = [
        e.message
        for e in sink.events
        if isinstance(e, LogMessage) and "could not verify the artifact" in e.message
    ]
    assert warns and "memory://m/model.ubj" in warns[0]


def test_rollback_auto_needs_integer_versions() -> None:
    reg = StubRegistry(champion=version(number="abc"))
    with pytest.raises(StateError, match="not an integer"):
        rollback_model(reg, name="m")


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


# -- mbt rollback (CLI) -------------------------------------------------------------


def _seed_two_versions(demo_project: Path) -> Path:
    """v1 (archived, gated) below v2 (the current production champion)."""
    registry_dir = demo_project / "target" / "fake_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    # a serviceable artifact: the rollback target's FILE must really exist,
    # because F12's head probe refuses a dangling reference
    artifact_file = registry_dir / "model.ubj"
    artifact_file.write_bytes(b"m")
    path = registry_dir / "churn_model.json"
    path.write_text(
        json.dumps(
            [
                {
                    "version": "1",
                    "stage": "archived",
                    "artifact": {
                        "uri": f"file://{artifact_file}",
                        "format": "xgboost_ubj",
                        "content_hash": "sha256:a",
                        "size_bytes": 1,
                    },
                    "tags": {"mbt.gates_passed": "true"},
                },
                {
                    "version": "2",
                    "stage": "production",
                    "artifact": None,
                    "tags": {"mbt.gates_passed": "true"},
                },
            ]
        )
    )
    return path


def test_rollback_cli_reverts_the_production_champion(demo_project: Path) -> None:
    registry_file = _seed_two_versions(demo_project)
    result = invoke(["rollback", "--model", "churn_model", "--project-dir", str(demo_project)])
    assert result.exit_code == 0, debug(result)
    assert "rolled back" in result.output
    stages = {e["version"]: e["stage"] for e in json.loads(registry_file.read_text())}
    assert stages["1"] == "production" and stages["2"] == "archived"  # champion moved down


def test_rollback_cli_requires_model(demo_project: Path) -> None:
    result = invoke(["rollback", "--project-dir", str(demo_project)])
    assert result.exit_code == 1, debug(result)
    assert "needs --model" in result.stderr
