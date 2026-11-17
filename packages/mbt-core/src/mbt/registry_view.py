"""``mbt registry ls``: what is in the model registry, read-only (FEEDBACK v7 D-3).

``mbt promote`` and ``mbt rollback`` act on registry versions, and until this
nothing in the CLI could show them: ``promote --version 999`` answered "list
versions in your registry UI", and the rollback runbook's "Verify" step had no
command to verify with. A version's stage and what it recorded (gates, the
after-test check) are exactly what a person deciding to promote or roll back
needs, so they are one table here, decoded through the champion record rather
than by tag spelling.
"""

from dataclasses import dataclass
from typing import Any

from mbt.exceptions import ConfigError
from mbt_adapter_base import ModelVersion
from mbt_adapter_base.capabilities import Capability, supports
from mbt_adapter_base.champion import GATES_PASSED, AfterTestVerdict


@dataclass(frozen=True)
class VersionRow:
    model: str
    version: str
    #: ``staging``, ``production`` or ``archived``; ``None`` for a version no
    #: stage points at.
    stage: str | None
    #: ``passed``, ``failed``, or ``not recorded`` for a version registered
    #: before gates were recorded or by something other than mbt.
    gates: str
    #: ``passed``, ``failed``, ``nothing mature``, or ``none`` when no
    #: after-test check has judged it (ADR-30).
    after_test: str
    #: The artifact's content hash, shortened; empty when the registry has none.
    artifact: str


def _gates(version: ModelVersion) -> str:
    recorded = version.tags.get(GATES_PASSED)
    if recorded == "true":
        return "passed"
    if recorded == "false":
        return "failed"
    return "not recorded"


def _after_test(version: ModelVersion) -> str:
    verdict = AfterTestVerdict.unpack(version.tags)
    if verdict is None:
        return "none"
    word = {"true": "passed", "false": "failed", "not_gated": "nothing mature"}.get(
        verdict.passed, verdict.passed
    )
    return f"{word} @ {verdict.anchor}" if verdict.anchor != "?" else word


def _row(version: ModelVersion) -> VersionRow:
    digest = version.artifact.content_hash if version.artifact else ""
    return VersionRow(
        model=version.name,
        version=version.version,
        stage=version.stage.value if version.stage else None,
        gates=_gates(version),
        after_test=_after_test(version),
        artifact=digest.removeprefix("sha256:")[:12],
    )


def list_versions(registry_adapter: Any, names: list[str]) -> list[VersionRow]:
    """Every registered version of each model in ``names``, newest first per model."""
    if not supports(registry_adapter, Capability.LIST_VERSIONS):
        raise ConfigError(
            f"registry adapter {type(registry_adapter).__name__!r} cannot list versions",
            hint="it does not declare Capability.LIST_VERSIONS; use the registry's own UI, or "
            "'mbt promote --version N' / 'mbt rollback --to-version N' with a known version",
        )
    return [_row(version) for name in names for version in registry_adapter.list_versions(name)]
