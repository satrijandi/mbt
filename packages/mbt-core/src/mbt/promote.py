"""``mbt promote``: gate-verified registry stage transitions (TSD §14.4, FR-REG-03)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from mbt.contracts import ModelVersion, Stage
from mbt.events import get_bus
from mbt.events.models import LogMessage, PromotionApplied
from mbt.exceptions import ConfigError, StateError


class PromotionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str  # registered model name
    to: Stage
    version: str | None = None  # default: latest in from_stage


class PromotionsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promotions: list[PromotionEntry]


@dataclass(frozen=True)
class PromotionOutcome:
    name: str
    version: str
    to_stage: Stage
    forced: bool


def load_promotions_file(path: Path) -> list[PromotionEntry]:
    if not path.is_file():
        raise ConfigError(f"promotions file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text()) or {}
        return PromotionsFile.model_validate(payload).promotions
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(
            f"invalid promotions file: {exc}",
            path=path,
            hint="expected: promotions: [{model: <name>, to: production, version: <n>}]",
        ) from exc


def _resolve_version(
    registry_adapter: Any, name: str, version: str | None, from_stage: Stage
) -> ModelVersion:
    if version is not None:
        resolved: ModelVersion | None = registry_adapter.get_version(name, version)
        if resolved is None:
            raise StateError(
                f"model {name!r} has no version {version!r}",
                hint="list versions in your registry UI, or omit --version for the latest",
            )
        return resolved
    champion: ModelVersion | None = registry_adapter.get_champion(name, from_stage)
    resolved = champion
    if resolved is None:
        raise StateError(
            f"model {name!r} has no version in stage {from_stage.value!r} to promote",
            hint="run 'mbt build' so a gated version lands in staging first",
        )
    return resolved


def promote_model(
    registry_adapter: Any,
    *,
    name: str,
    to_stage: Stage,
    version: str | None = None,
    from_stage: Stage = Stage.STAGING,
    force: bool = False,
) -> PromotionOutcome:
    """Resolve, verify recorded gate passes, transition (TSD §14.4)."""
    resolved = _resolve_version(registry_adapter, name, version, from_stage)
    gates_passed = resolved.tags.get("mbt.gates_passed") == "true"
    if not gates_passed:
        if not force:
            raise StateError(
                f"refusing to promote {name} v{resolved.version}: gates were not "
                "recorded as passed at registration",
                hint="fix the model until its gates pass, or override with --force",
            )
        get_bus().emit(
            LogMessage(
                level="warn",
                message=(
                    f"FORCED promotion of {name} v{resolved.version} without recorded "
                    "gate passes - this bypasses the quality contract"
                ),
            )
        )
    registry_adapter.transition(resolved, to_stage)
    get_bus().emit(
        PromotionApplied(
            name=name, version=resolved.version, to_stage=to_stage.value, forced=not gates_passed
        )
    )
    return PromotionOutcome(
        name=name, version=resolved.version, to_stage=to_stage, forced=not gates_passed
    )


def _last_known_good_below(registry_adapter: Any, name: str, champion_version: str) -> str:
    """The highest version below the champion that recorded passing gates.

    Probes downward via ``get_version`` (versions are sequential integers), so
    it needs no new registry-protocol method. Skips versions whose gates did not
    pass - the point of a rollback is to land on a known-good prior champion.
    """
    try:
        start = int(champion_version) - 1
    except ValueError as exc:  # non-integer version scheme: caller must be explicit
        raise StateError(
            f"cannot auto-detect a rollback target for {name!r} (version "
            f"{champion_version!r} is not an integer)",
            hint="pass --to-version to name the version to roll back to",
        ) from exc
    for candidate in range(start, 0, -1):
        prior = registry_adapter.get_version(name, str(candidate))
        if prior is not None and prior.tags.get("mbt.gates_passed") == "true":
            return str(candidate)
    raise StateError(
        f"model {name!r} has no earlier gated version below v{champion_version} to roll back to",
        hint="pass --to-version to name a specific version, or --force to bypass the gate record",
    )


def rollback_model(
    registry_adapter: Any,
    *,
    name: str,
    to_version: str | None = None,
    force: bool = False,
) -> PromotionOutcome:
    """Revert the production champion to a prior version (incident response).

    With ``to_version`` that exact version is re-promoted; otherwise the most
    recent version below the current champion that recorded passing gates (the
    last known good) is chosen, so an operator need not look up a version number
    mid-incident. Re-promotes through :func:`promote_model`, so a target that
    passed its gates promotes cleanly and one that did not still needs ``force``.
    """
    current = registry_adapter.get_champion(name, Stage.PRODUCTION)
    if current is None:
        raise StateError(
            f"model {name!r} has no production champion to roll back from",
            hint="rollback reverts the production alias; promote a version to production first",
        )
    target = (
        to_version
        if to_version is not None
        else _last_known_good_below(registry_adapter, name, current.version)
    )
    if str(target) == str(current.version):
        raise StateError(
            f"model {name!r} v{current.version} is already the production champion",
            hint="name an earlier version with --to-version",
        )
    # An incident-response rollback must land on a SERVICEABLE version: if the
    # target's artifact was aged out (e.g. by `mbt clean`, which keeps only
    # current champions + the latest run), moving the alias would "succeed" and
    # the NEXT `mbt score` would then die with 'no loadable artifact'. Refuse at
    # this command instead (F12). A missing target version falls through to
    # promote_model's clearer "no version" error.
    target_mv = registry_adapter.get_version(name, str(target))
    if target_mv is not None and target_mv.artifact is None:
        raise StateError(
            f"cannot roll back {name!r} to v{target}: that version has no loadable "
            "artifact (it may have been aged out by 'mbt clean')",
            hint="roll back to a version whose artifact still exists, or re-train it",
        )
    if target_mv is not None and target_mv.artifact is not None:
        # The ref surviving in the registry proves nothing: the FILE behind it
        # can be gone (`mbt clean`, a bucket lifecycle rule). Head-probe it so
        # the failure lands at this command, not at the next `mbt score` (F12).
        from mbt.storage import artifact_exists

        exists = artifact_exists(target_mv.artifact)
        if exists is False:
            raise StateError(
                f"cannot roll back {name!r} to v{target}: its artifact at "
                f"{target_mv.artifact.uri} no longer exists (aged out by "
                "'mbt clean' or a bucket lifecycle rule)",
                hint="roll back to a version whose artifact still exists, or re-train it",
            )
        if exists is None:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    message=(
                        f"rollback target v{target}: could not verify the artifact at "
                        f"{target_mv.artifact.uri} (unrecognized scheme or missing "
                        "extra); proceeding - the next 'mbt score' fails if it is gone"
                    ),
                )
            )
    outcome = promote_model(
        registry_adapter, name=name, to_stage=Stage.PRODUCTION, version=str(target), force=force
    )
    get_bus().emit(
        LogMessage(
            level="warn",
            message=(
                f"ROLLBACK: {name} production reverted from "
                f"v{current.version} to v{outcome.version}"
            ),
        )
    )
    return outcome
