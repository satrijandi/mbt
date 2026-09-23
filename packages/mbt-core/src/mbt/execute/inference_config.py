"""The inference config a champion carries with it (ADR-28).

Every training job exports one of these beside the model artifact, and
registration pins it on the model version as ``mbt.inference_config_uri``.
``mbt score`` reads the model's spec back from it rather than recompiling the
spec out of the local manifest, so a scoring run is driven by the champion it
actually loaded instead of by whatever the working tree says today.

``spec`` is the node's rendered config verbatim - the same mapping
``ModelSpec.model_validate`` consumes and the same one ``config_hash`` is
taken over - so reconstruction at score time is exact and independently
checkable. Everything under ``resolved`` is what the manifest cannot answer:
``features.include: ["*"]`` does not say which columns the model was fit on,
in which order.

Not in here on purpose: ``hooks.py``'s source. Hooks are arbitrary Python,
and mbt executes the git-tracked file from the project checkout rather than
code fetched from a registry (the hash below is what proves the two agree).
The training run logs the source to the tracker as a readable record.
"""

from typing import Any

from mbt_adapter_base import ArtifactRef, ManifestNode
from mbt_adapter_base.champion import GIT_COMMIT, MANIFEST_HASH, SNAPSHOT_ID

#: Bumped when a consumer would read an older document wrongly. Additive keys
#: do not bump it; a rename or a changed meaning does.
SCHEMA_VERSION = 1

#: Metric names whose values are deployable score cutoffs rather than
#: qualities (R2-5); a scoring node names one in ``decision_threshold``.
_OPERATING_POINT_PREFIXES = ("threshold_at_precision_", "threshold_at_recall_")


def operating_points(metrics: dict[str, float]) -> dict[str, float]:
    """The subset of a metric set that names a deployable cutoff (R2-5)."""
    return {
        name: float(value)
        for name, value in metrics.items()
        if name.startswith(_OPERATING_POINT_PREFIXES)
    }


def build_inference_config(
    *,
    node: ManifestNode,
    project: str,
    run_id: str,
    feature_columns: list[str],
    metrics: dict[str, float],
    artifact: ArtifactRef,
    baseline: ArtifactRef | None,
    meta: dict[str, str],
    hyperparameters: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    report: Any = None,
) -> dict[str, Any]:
    """Assemble the document exported next to the model artifact.

    ``meta`` is the job's tracking metadata, which already carries the
    manifest hash and git commit; reading identity from there keeps one source
    for it rather than threading the manifest into the job a second time.
    """
    spec = node.config
    declared = spec.get("features")
    features: dict[str, Any] = declared if isinstance(declared, dict) else {}
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "model": node.name,
        "unique_id": node.unique_id,
        "trained_at": run_id,
        "spec": spec,
        "resolved": {
            # The exact column set and order the adapter was fit on, taken from
            # the same transformed handle the monitoring baseline is built from,
            # so the two describe one population.
            "feature_columns": list(feature_columns),
            "target": spec.get("target"),
            "task": node.task,
            "adapter": node.adapter,
            "seed": node.seed,
            "calibration": spec.get("calibration"),
            "categorical": features.get("categorical"),
            "transforms": features.get("transforms"),
            "monotonic": features.get("monotonic"),
            "operating_points": operating_points(metrics),
        },
        "hooks": {
            "path": node.hooks_path,
            "hash": node.hooks_hash,
        },
        "identity": {
            "config_hash": node.config_hash,
            "input_hash": node.input_hash,
            "manifest_hash": meta.get(MANIFEST_HASH, ""),
            "snapshot_id": meta.get(SNAPSHOT_ID, ""),
            "git_commit": meta.get(GIT_COMMIT, ""),
        },
        "artifact": {
            "uri": artifact.uri,
            "format": artifact.format,
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
        },
        "baseline_uri": baseline.uri if baseline is not None else None,
    }
    # Additive (ADR-30), so no schema bump: the values AUTO resolution and
    # tuning settled on, which `spec` still holds as sentinels and search
    # spaces; the dataset side a serving system needs to rebuild the input;
    # and what the training report measured, which the pre-deploy check
    # compares a rebuilt test window against.
    if hyperparameters is not None:
        document["resolved"]["hyperparameters"] = hyperparameters
    if dataset is not None:
        document["dataset"] = dataset
    if report is not None:
        document["report"] = {
            "uri": report.report_uri,
            "reference_rows": report.reference_rows,
            "out_of_time_rows": report.out_of_time_rows,
        }
    return document
