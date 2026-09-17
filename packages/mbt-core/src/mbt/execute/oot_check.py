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
from mbt.contracts import OUT_OF_TIME_SPLIT, ManifestNode, ModelSpec, ModelVersion
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import StateError
from mbt.execute.runners import (
    DatasetRunner,
    ExecutionContext,
    ModelRunner,
    after_test_tags,
    after_test_verdict,
    check_hooks_parity,
    gate_failure_summary,
    read_inference_config,
    run_with_lifecycle,
)
from mbt.quality.gates import all_gates_passed
from mbt.quality.monitors import all_monitors_passed, evaluate_stability


def _instant(iso: str) -> datetime:
    from mbt_adapter_base.reporting import parse_window_bound

    return parse_window_bound(iso)


def recorded_training(ctx: ExecutionContext, version: ModelVersion, uid: str) -> dict[str, Any]:
    """The version's inference config, refusing what the check cannot use."""
    name = f"{version.name} v{version.version}"
    if not version.tags.get("mbt.inference_config_uri"):
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


def _pin_windows(
    ctx: ExecutionContext, model_uid: str, record: dict[str, Any]
) -> tuple[str, list[str]]:
    """Point this invocation's dataset and model nodes at the recorded windows.

    Only the in-memory manifest changes - ``target/manifest.json`` keeps what
    compile wrote - and the pinned windows give the materialization its own
    cache key, so nothing trained elsewhere reads these rows.
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
    dataset_uid, _ = _pin_windows(ctx, model_uid, record)

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
    record = document["dataset"]
    _check_reference(ctx, str(record["unique_id"]), record, model_uid)
    check_hooks_parity(version, node, model_uid)
    spec = ModelSpec.model_validate(document["spec"])
    resolved = document.get("resolved") or {}
    runner = ModelRunner(ctx)
    metric_specs = runner._metric_specs(spec, node)
    job = runner._assemble_job(node, spec, metric_specs, None, mode="oot_check").model_copy(
        update={
            "artifact": version.artifact,
            "champion_spec": document["spec"],
            "champion_feature_columns": resolved.get("feature_columns"),
            "tracking_run_id": version.tags.get("mbt.tracking_run_id") or None,
        }
    )
    job_result = ctx.run_job(job)
    runner._upload_log(node, job.tracking_run_id, name=f"evaluate-{ctx.run_id}.log")
    if job_result.status == "error" or job_result.report is None:
        return NodeResult(
            unique_id=model_uid,
            status="error",
            message=job_result.error or "the check returned no report",
        )
    summary = job_result.report
    gates = []
    stability = []
    passed = True
    if apply_gates:
        gates = runner._gate_results(spec, node, job_result, None, metric_specs, out_of_time="only")
        stability = evaluate_stability(spec.evaluation.stability, summary, resource=model_uid)
        passed = all_gates_passed(gates) and all_monitors_passed(stability)
        verdict = after_test_verdict(spec, gates, stability) or "not_gated"
        _record(ctx, version, job.tracking_run_id, verdict, summary.report_uri, model_uid)
    return NodeResult(
        unique_id=model_uid,
        status="success" if passed else "gate_failed",
        metrics=dict(job_result.metrics.metrics) if job_result.metrics else {},
        gates=gates,
        monitors=stability,
        artifact=version.artifact,
        tracking_run_id=job.tracking_run_id,
        message=None if passed else gate_failure_summary(gates, stability),
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
    tags["mbt.oot_check.run_id"] = ctx.run_id
    if report_uri:
        tags["mbt.oot_check.report_uri"] = report_uri
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
