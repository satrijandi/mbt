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
        resolved = registry_adapter.get_version(name, version)
        if resolved is None:
            raise StateError(
                f"model {name!r} has no version {version!r}",
                hint="list versions in your registry UI, or omit --version for the latest",
            )
        return resolved
    resolved = registry_adapter.get_champion(name, from_stage)
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
