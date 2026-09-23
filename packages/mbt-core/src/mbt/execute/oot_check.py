"""``mbt evaluate --out-of-time``: the pre-deploy check (ADR-30).

A version is trained on a test window that is often months old by the day it
ships. This re-runs the training report for that registered version against
everything since: the version's own recorded test window is the reference,
and the after-test window runs from its end to this invocation's anchor.
The report is appended to the version's training run, and - with
``--gates`` - its after-test gates and stability gates are judged and the
verdict is recorded on the version, where ``mbt promote`` reads it.
"""

from datetime import datetime
from typing import Any

from mbt.artifacts.run_results import NodeResult
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import StateError
from mbt.execute.runners import (
    DatasetRunner,
    ExecutionContext,
    ModelRunner,
    check_hooks_parity,
    read_inference_config,
    run_with_lifecycle,
)
from mbt.quality.judgement import Judgement, after_test_tags, judge
from mbt_adapter_base import (
    OUT_OF_TIME_SPLIT,
    ManifestNode,
    ModelSpec,
    ModelVersion,
)
from mbt_adapter_base.champion import (
    OOT_CHECK_REPORT_URI,
    OOT_CHECK_RUN_ID,
    TRACKING_RUN_ID,
    has_inference_config,
)


def _instant(iso: str) -> datetime:
    from mbt_adapter_base.reporting import parse_window_bound

    return parse_window_bound(iso)


def recorded_training(ctx: ExecutionContext, version: ModelVersion, uid: str) -> dict[str, Any]:
    """The version's inference config, refusing what the check cannot use."""
    name = f"{version.name} v{version.version}"
    if not has_inference_config(version.tags):
        raise StateError(
            f"{name} predates inference-config export, so its training windows are unknown",
            resource=uid,
            hint="retrain and register the model; the check needs the recorded test window",
        )
    document = read_inference_config(ctx, version)
    record = document.get("dataset")
    if not isinstance(record, dict):
        raise StateError(
            f"{name} was registered before mbt recorded its dataset windows (ADR-30)",
            resource=uid,
            hint="retrain and register the model to run the pre-deploy check on it",
        )
    if record.get("split_strategy") != "temporal":
        raise StateError(
            f"{name} was trained on a random split; 'after the test window' has no meaning",
            resource=uid,
            hint="the pre-deploy check needs a temporal split",
        )
    return document


def pin_check_windows(
    ctx: ExecutionContext, model_uid: str, record: dict[str, Any]
) -> tuple[str, list[str]]:
    """Point this invocation's dataset and model nodes at the recorded windows.

    Only the in-memory manifest changes - ``target/manifest.json`` keeps what
    compile wrote - and the pinned windows give the materialization its own
    cache key, so nothing trained elsewhere reads these rows.

    **Ordering.** This MUST run before ``DatasetRunner.run(dataset_uid)``,
    because the value it writes is read back four frames later by
    ``DatasetRunner._materialize``. Nothing enforced that and nothing named it
    (C-3); it is public and named now, it returns the uid the caller must build,
    and ``test_oot_check_unit.py`` asserts what it pins - which is the seam at
    which "given a version and a manifest, what windows get pinned" can be
    asked at all.
    """
    dataset_uid = str(record["unique_id"])
    if dataset_uid not in ctx.manifest.nodes:
        raise StateError(
            f"the version was trained on {dataset_uid}, which this project no longer has",
            resource=model_uid,
            hint="check out the commit the version was trained from",
        )
    test = list(record["windows"]["test"])
    anchor = ctx.manifest.metadata.anchor
    if _instant(anchor) <= _instant(test[1]):
        raise StateError(
            f"nothing to check yet: the anchor {anchor} is not after the version's test "
            f"window, which ends at {test[1]}",
            resource=model_uid,
            hint="run the check later, or pass a later --anchor",
        )
    dataset_node = ctx.manifest.nodes[dataset_uid]
    if dataset_node.config_hash != record.get("config_hash"):
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=model_uid,
                message=(
                    f"{dataset_uid} has changed since the version was trained (config hash "
                    f"{str(record.get('config_hash'))[:19]} then, {dataset_node.config_hash[:19]} "
                    "now); the check reads the dataset as it is declared today"
                ),
            )
        )
    windows = {"test": test, OUT_OF_TIME_SPLIT: [test[1], anchor]}
    ctx.manifest.nodes[dataset_uid] = dataset_node.model_copy(
        update={"resolved": {**dataset_node.resolved, "windows": windows}}
    )
    model_node = ctx.manifest.nodes[model_uid]
    resolved = {k: v for k, v in model_node.resolved.items() if k != "test_window"}
    if record.get("test_window"):
        resolved["test_window"] = list(record["test_window"])
    ctx.manifest.nodes[model_uid] = model_node.model_copy(update={"resolved": resolved})
    return dataset_uid, test


def _check_reference(
    ctx: ExecutionContext, dataset_uid: str, record: dict[str, Any], uid: str
) -> None:
    """Warn when the rebuilt test window is not the one the version saw."""
    recorded = (record.get("row_counts") or {}).get("test")
    handle = ctx.dataset_handle(dataset_uid)
    rebuilt = handle.profile().n_rows.get("test")
    if recorded is not None and rebuilt is not None and int(recorded) != int(rebuilt):
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    f"the version's test window now holds {rebuilt} rows, not the {recorded} "
                    "it was evaluated on; the source data or the dataset's declaration "
                    "changed since, so the reference is not the one the version was judged "
                    "against"
                ),
            )
        )


def run_oot_check(
    ctx: ExecutionContext,
    model_uid: str,
    version: ModelVersion,
    *,
    apply_gates: bool,
) -> list[NodeResult]:
    """Build the pinned dataset, run the check job, judge, record."""
    document = recorded_training(ctx, version, model_uid)
    record = document["dataset"]
    # Pin BEFORE the dataset build: the build reads what this writes.
    dataset_uid, _ = pin_check_windows(ctx, model_uid, record)

    results = [DatasetRunner(ctx).run(dataset_uid)]
    if results[0].status == "error":
        results.append(
            NodeResult(
                unique_id=model_uid,
                status="skipped",
                message=f"upstream {dataset_uid} error",
            )
        )
        return results

    results.append(
        run_with_lifecycle(
            ctx,
            model_uid,
            "model",
            lambda: _check(ctx, model_uid, version, document, apply_gates=apply_gates),
        )
    )
    return results


def _check(
    ctx: ExecutionContext,
    model_uid: str,
    version: ModelVersion,
    document: dict[str, Any],
    *,
    apply_gates: bool,
) -> NodeResult:
    node: ManifestNode = ctx.manifest.nodes[model_uid]
    if version.artifact is None:
        # Surfaced by C-3's typed seam: the old path patched ``artifact`` onto
        # the job through a ``model_copy`` with an untyped dict, so a version
        # with no loadable artifact produced a job with ``artifact=None`` and
        # failed later, somewhere else. Same stance as the scoring path: a
        # version that exists but cannot load is an error (ADR-10).
        raise StateError(
            f"{version.name} v{version.version} has no loadable artifact reference to check",
            resource=model_uid,
            hint="re-register the version; the check has to load the model it judges",
        )
    record = document["dataset"]
    _check_reference(ctx, str(record["unique_id"]), record, model_uid)
    check_hooks_parity(version, node, model_uid)
    spec = ModelSpec.model_validate(document["spec"])
    resolved = document.get("resolved") or {}
    runner = ModelRunner(ctx)
    # Public seams, not four private methods and a patch of the returned job (C-3).
    job, metric_specs = runner.check_job(
        node,
        spec,
        artifact=version.artifact,
        champion_spec=document["spec"],
        champion_feature_columns=resolved.get("feature_columns"),
        tracking_run_id=version.tags.get(TRACKING_RUN_ID) or None,
    )
    job_result = ctx.run_job(job)
    runner.upload_run_log(node, job.tracking_run_id, name=f"evaluate-{ctx.run_id}.log")
    if job_result.status == "error" or job_result.report is None:
        return NodeResult(
            unique_id=model_uid,
            status="error",
            message=job_result.error or "the check returned no report",
        )
    summary = job_result.report
    outcome: Judgement | None = None
    if apply_gates:
        gates = runner.after_test_gates(spec, node, job_result, metric_specs)
        # The same judgement the training path makes (C-1), over the after-test
        # gates only - which is the one thing that genuinely differs here.
        outcome = judge(spec, gates, summary, resource=model_uid)
        _record(
            ctx,
            version,
            job.tracking_run_id,
            outcome.verdict or "not_gated",
            summary.report_uri,
            model_uid,
        )
    return NodeResult(
        unique_id=model_uid,
        status="success" if (outcome is None or outcome.passed) else "gate_failed",
        metrics=dict(job_result.metrics.metrics) if job_result.metrics else {},
        gates=outcome.gates if outcome else [],
        monitors=outcome.stability.results if outcome else [],
        artifact=version.artifact,
        tracking_run_id=job.tracking_run_id,
        message=outcome.failure_summary if outcome else None,
    )


def _record(
    ctx: ExecutionContext,
    version: ModelVersion,
    tracking_run_id: str | None,
    verdict: str,
    report_uri: str | None,
    uid: str,
) -> None:
    """The verdict on the version (what ``mbt promote`` reads) and its run."""
    tags = after_test_tags(verdict, ctx.manifest.metadata.anchor, "evaluate")
    tags[OOT_CHECK_RUN_ID] = ctx.run_id
    if report_uri:
        tags[OOT_CHECK_REPORT_URI] = report_uri
    registry = ctx.registry_adapter()
    if hasattr(registry, "set_version_tags"):
        registry.set_version_tags(version.name, version.version, tags)
    else:
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    "the registry adapter cannot tag versions (no set_version_tags); the "
                    "verdict is recorded only on the tracking run and in run_results"
                ),
            )
        )
    if tracking_run_id:
        try:
            tracking = ctx.tracking()
            tracking.log(tracking.resume(tracking_run_id), tags=tags)
        except Exception as exc:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=uid,
                    message=f"could not record the check on the training run: {exc}",
                )
            )
    outcome = _VERDICT_WORDS.get(verdict, verdict)
    get_bus().emit(
        LogMessage(
            level="warn" if verdict == "false" else "info",
            unique_id=uid,
            message=f"pre-deploy check on {version.name} v{version.version} {outcome}; "
            f"recorded as mbt.oot_check.passed={verdict}",
        )
    )


_VERDICT_WORDS = {
    "true": "passed",
    "false": "FAILED",
    "not_gated": "judged nothing (no after-test gate had a mature cell)",
}
