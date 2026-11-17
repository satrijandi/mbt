"""Reading a champion's recorded inference config, and the parity checks on it.

Shared by the model runner (which carries a champion's config forward) and the
scoring runner (which scores with it).
"""

import json
from typing import Any, NamedTuple

from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import StateError
from mbt.execute.runners.context import ExecutionContext
from mbt.runtime import (
    resolve_artifact_store_uri,
)
from mbt.storage import artifact_store_for
from mbt_adapter_base import (
    ManifestNode,
    ModelVersion,
)
from mbt_adapter_base.champion import (
    HOOKS_HASH,
    INFERENCE_CONFIG,
    unpack_artifact,
)


class ChampionConfig(NamedTuple):
    """What `mbt score` reads back off the champion's inference config (ADR-28).

    ``spec`` is authoritative over the working tree's model node, and
    ``feature_columns`` over re-running the include/exclude globs against
    whatever the batch holds (ADR-29). Both are None for a champion registered
    before mbt exported an inference config.
    """

    spec: dict[str, Any] | None
    feature_columns: list[str] | None


def check_hooks_parity(version: ModelVersion, model_node: ManifestNode, uid: str) -> None:
    """A registered version must have been trained with the hooks this run
    will apply; silent feature skew is worse than a hard stop (ADR-20)."""
    registered = version.tags.get(HOOKS_HASH)
    if registered is None:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    "champion predates hooks-parity registration (no "
                    "mbt.hooks_hash tag); cannot verify that scoring applies "
                    "the hooks the champion was trained with"
                ),
            )
        )
        return
    if registered != (model_node.hooks_hash or ""):
        raise StateError(
            "the champion was trained with a different hooks.py than the "
            "current project's (mbt.hooks_hash mismatch)",
            resource=uid,
            hint="retrain and promote, or check out the commit the champion was built from",
        )


def read_inference_config(ctx: ExecutionContext, version: ModelVersion) -> dict[str, Any]:
    """The inference config a version was registered with (ADR-28), fetched
    and content-verified from the artifact store.

    The obligation "callers check the tag exists first" used to be held in this
    docstring and honoured by one of its callers (A-5). It is
    ``has_inference_config`` now, and this raises rather than KeyError-ing when
    a caller forgets.
    """
    ref = unpack_artifact(INFERENCE_CONFIG, version.tags)
    if ref is None:
        raise StateError(
            "the champion carries no inference config to read",
            hint="callers check the tag exists first; see has_inference_config",
        )
    store = artifact_store_for(
        resolve_artifact_store_uri(ctx.profiles.target.artifact_store, ctx.project_dir)
    )
    document: dict[str, Any] = json.loads(store.fetch(ref).read_text())
    return document
