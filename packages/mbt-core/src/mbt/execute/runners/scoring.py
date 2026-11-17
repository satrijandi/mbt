"""The scoring runner: resolve the champion, build the batch, score, monitor shift."""

import hashlib
from typing import Any

from mbt.artifacts.run_results import (
    NodeResult,
    TestResultEntry,
)
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import AdapterError, ConfigError, StateError
from mbt.execute.runners.champion_config import (
    ChampionConfig,
    check_hooks_parity,
    read_inference_config,
)
from mbt.execute.runners.context import (
    BuildContext,
    ExecutionContext,
    materialization_key,
    run_with_lifecycle,
    source_access,
)
from mbt.quality.checks import run_scoring_checks
from mbt.quality.monitors import all_monitors_passed, evaluate_monitors
from mbt.runtime import (
    resolve_artifact_store_uri,
)
from mbt.utils import canonical_json
from mbt_adapter_base import (
    ArtifactRef,
    ManifestNode,
    ModelSpec,
    ModelVersion,
    ScoringOutputSpec,
    ScoringSpec,
    TrainingJob,
)
from mbt_adapter_base.champion import (
    BASELINE,
    has_inference_config,
    unpack_artifact,
)


def scoring_run_key(node: ManifestNode, model_version: str) -> str:
    """Prediction-run idempotency key (ADR-21): same manifest + same champion
    re-scores overwrite cleanly; new data, window, or champion partitions
    fresh. The scoring analog of ``materialization_key``."""
    digest = hashlib.sha256()
    digest.update(node.input_hash.encode())
    digest.update(canonical_json(node.resolved.get("windows", {})).encode())
    digest.update(model_version.encode())
    return digest.hexdigest()[:16]


class ScoringRunner:
    """Coordinator side of a scoring run (ADR-20): jobs compute, core compares.

    The champion is resolved from the registry at RUN time by stage alias -
    promotions are registry state, deliberately outside node identity
    (ADR-5), so scheduled scoring picks up a new champion on its next run
    without a spec edit. The resolved version lands in run_results, the
    tracking tags, and the prediction sidecar.
    """

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "scoring", lambda: self._run_inner(uid, node))

    # -- helpers -------------------------------------------------------------

    def _champion(self, spec: ScoringSpec, model_spec: ModelSpec, uid: str) -> ModelVersion:
        registry_name = model_spec.registration.name if model_spec.registration else model_spec.name
        champion: ModelVersion | None = self.ctx.registry_adapter().get_champion(
            registry_name, spec.stage
        )
        if champion is None:
            # Unlike a missing gate comparator (ADR-10 WARN), nothing to
            # score WITH is an operational failure.
            raise StateError(
                f"no champion of {registry_name!r} in stage {spec.stage.value!r} to score with",
                resource=uid,
                hint="train and promote the model first (mbt build, then mbt promote)",
            )
        if champion.artifact is None:
            raise AdapterError(
                f"champion {registry_name} v{champion.version} in {spec.stage.value!r} has no "
                "loadable artifact reference",
                resource=uid,
                hint="a champion that exists but cannot load is an error (ADR-10)",
            )
        return champion

    def _check_hooks_parity(
        self, champion: ModelVersion, model_node: ManifestNode, uid: str
    ) -> None:
        check_hooks_parity(champion, model_node, uid)

    def _baseline_ref(self, version: ModelVersion) -> ArtifactRef | None:
        return unpack_artifact(BASELINE, version.tags)

    def _champion_config(
        self, champion: ModelVersion, model_node: ManifestNode, uid: str
    ) -> "ChampionConfig":
        """The spec and feature columns the champion was trained with (ADR-28).

        Scoring follows the champion the registry resolved, not the working
        tree: a promotion is deliberately outside node identity (ADR-5), so
        the two legitimately diverge between a retrain and a promote. Reading
        the spec from the champion is what makes the scoring run use the
        features, treatment and categoricals the model was actually fit on.

        Both fields are None for a champion registered before mbt exported an
        inference config, which leaves the caller on the local manifest exactly
        as before - the same shape ADR-21 gave a champion registered before
        baselines existed.
        """
        if not has_inference_config(champion.tags):
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=uid,
                    message=(
                        f"champion v{champion.version} predates inference-config export "
                        "(no mbt.inference_config_uri tag); scoring falls back to the "
                        "project's current model spec, which may not be the one it was "
                        "trained with - retrain and promote to close this"
                    ),
                )
            )
            return ChampionConfig(None, None)
        document = read_inference_config(self.ctx, champion)
        spec = document.get("spec")
        if not isinstance(spec, dict):
            raise StateError(
                f"champion v{champion.version} inference config has no model spec",
                resource=uid,
                hint="the recorded inference config is malformed; retrain and promote",
            )
        recorded = str(document.get("identity", {}).get("config_hash", ""))
        if recorded and recorded != model_node.config_hash:
            # Expected whenever a spec edit has been merged but not yet
            # promoted; scoring uses the champion's, and says which.
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=uid,
                    message=(
                        f"scoring with champion v{champion.version}, whose spec differs from "
                        f"the project's ({recorded[:12]} vs {model_node.config_hash[:12]}); "
                        "the champion's spec is authoritative - promote the retrained "
                        "model to close the gap"
                    ),
                )
            )
        resolved = document.get("resolved")
        columns = resolved.get("feature_columns") if isinstance(resolved, dict) else None
        # An EMPTY recorded list is a MISSING pin, not a pin of zero features
        # (C-2): the write side and the read side disagreed about what empty
        # meant, and `_apply_pin` treats any non-None pin as authoritative.
        pinned = list(columns) if isinstance(columns, list) and columns else None
        return ChampionConfig(spec, pinned)

    def _materialize_input(self, node: ManifestNode, spec: ScoringSpec) -> Any:
        ctx = self.ctx
        sample_fraction = ctx.sample_fraction()
        key = materialization_key(node, sample_fraction)
        output_dir = ctx.layout.scoring_input_dir(node.name, key)
        # No cache reuse for scoring inputs: a scoring batch is expected-mutable
        # (fresh unlabeled rows, or the labels `mbt monitor` reads through this
        # same path, change every run), and under --manifest the key is fully
        # pinned (input_hash + windows never move), so a warm `_SUCCESS` here
        # would serve a STALE batch (F4). build_scoring_input wipes and rebuilds
        # this one cheap batch every call; datasets, which are immutable by
        # snapshot, still cache in DatasetRunner._materialize.
        input_uids = [spec.input.source]
        source_tables = {
            uid: ctx.manifest.sources[uid].config
            for uid in input_uids
            if uid in ctx.manifest.sources
        }
        if set(source_tables) != set(input_uids):
            missing = sorted(set(input_uids) - set(source_tables))
            raise ConfigError(
                f"scoring input source(s) missing from the manifest: {', '.join(missing)}",
                resource=node.unique_id,
                hint="recompile: the manifest and spec disagree",
            )
        windows = {
            split: (bounds[0], bounds[1])
            for split, bounds in node.resolved.get("windows", {}).items()
        }
        build_ctx = BuildContext(
            node=node,
            source=source_tables[spec.input.source],
            source_tables=source_tables,
            resolved_windows=windows,
            sample_fraction=sample_fraction,
            deep_snapshot=ctx.manifest.metadata.deep_snapshot,
            output_dir=output_dir,
            events=get_bus(),
            build_parallelism=ctx.threads,
        )
        return ctx.data_adapter.build_scoring_input(spec.input, build_ctx)

    @staticmethod
    def _resolve_operating_point(
        output: ScoringOutputSpec, champion: ModelVersion, uid: str
    ) -> ScoringOutputSpec:
        """A string ``decision_threshold`` names a champion operating-point
        metric; resolve it to the concrete cutoff from the champion's registered
        tags (R2-5), so the deployed decision rule tracks the promoted model
        rather than a constant copied out of the model card by hand."""
        name = output.decision_threshold
        if not isinstance(name, str):
            return output
        recorded = champion.tags.get(f"mbt.operating_point.{name}")
        if recorded is None:
            raise StateError(
                f"scoring decision_threshold {name!r} is not a recorded operating point of "
                f"champion v{champion.version}",
                resource=uid,
                hint=f"add {name!r} to the model's evaluation.metrics and retrain/promote, "
                "or set a numeric decision_threshold",
            )
        return output.model_copy(update={"decision_threshold": float(recorded)})

    def _assemble_job(
        self,
        node: ManifestNode,
        model_node: ManifestNode,
        spec: ScoringSpec,
        champion: ModelVersion,
        baseline: ArtifactRef | None,
        champion_config: "ChampionConfig",
        handle: Any,
    ) -> TrainingJob:
        ctx = self.ctx
        meta = ctx.manifest.metadata
        # No `tracking` ref and no tracking_meta: a scoring job opens no
        # tracking run (ADR-28), so handing it a tracker would only give it a
        # backend it must not use.
        return TrainingJob(
            mode="score",
            run_id=ctx.run_id,
            project_dir=str(ctx.project_dir),
            target_name=meta.target,
            project=meta.project_name,
            anchor=meta.anchor,
            node=node,
            model_node=model_node,
            champion_spec=champion_config.spec,
            champion_feature_columns=champion_config.feature_columns,
            dataset=handle.locator(),
            data=ctx.raw_adapter_ref("data"),
            artifact=champion.artifact,
            baseline=baseline,
            output=self._resolve_operating_point(spec.output, champion, node.unique_id),
            model_version=champion.version,
            run_key=scoring_run_key(node, champion.version),
            artifact_store=resolve_artifact_store_uri(
                ctx.profiles.target.artifact_store, ctx.project_dir
            ),
            required_env=list(ctx.profiles.required_env),
            vars=ctx.job_safe_vars(),
        )

    # -- main path -------------------------------------------------------------

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        ctx = self.ctx
        spec = ScoringSpec.model_validate(node.config)
        model_uid = next(d for d in node.depends_on if d.startswith("model."))
        model_node = ctx.manifest.nodes.get(model_uid)
        if model_node is None:
            raise ConfigError(
                f"scoring model {model_uid!r} missing from the manifest",
                resource=uid,
                hint="recompile: the manifest and spec disagree",
            )
        model_spec = ModelSpec.model_validate(model_node.config)

        champion = self._champion(spec, model_spec, uid)
        # Still the local hooks file, still a hard fail: hooks are arbitrary
        # Python, so mbt runs the git-tracked one and refuses when it is not
        # the one the champion was trained with (ADR-20, ADR-28).
        self._check_hooks_parity(champion, model_node, uid)
        baseline = self._baseline_ref(champion)
        champion_config = self._champion_config(champion, model_node, uid)

        handle = self._materialize_input(node, spec)

        checks = run_scoring_checks(
            spec, handle, node.resolved, resource=uid, sources=source_access(self.ctx, node)
        )
        failed = [t for t in checks if not t.passed]
        if failed:
            # Never score on bad input: the job is skipped entirely.
            return NodeResult(
                unique_id=uid,
                status="test_failed",
                tests=[TestResultEntry(**t.model_dump()) for t in checks],
                message=(
                    f"{len(failed)} input check failure(s), scoring skipped: "
                    + "; ".join(f"{t.name}: {t.message}" for t in failed)
                ),
            )

        job = self._assemble_job(
            node, model_node, spec, champion, baseline, champion_config, handle
        )
        job_result = ctx.run_job(job)
        if job_result.status == "error":
            return NodeResult(
                unique_id=uid, status="error", message=job_result.error or "scoring job failed"
            )

        monitors = evaluate_monitors(spec.monitors, job_result.monitor_stats, resource=uid)
        passed = all_monitors_passed(monitors)
        rows = float(job_result.predictions.row_count) if job_result.predictions else 0.0
        failures = [m.message for m in monitors if not m.passed and m.message]
        return NodeResult(
            unique_id=uid,
            status="success" if passed else "monitor_failed",
            metrics={"rows_scored": rows},
            monitors=monitors,
            tests=[TestResultEntry(**t.model_dump()) for t in checks],
            tracking_run_id=job_result.tracking_run_id,
            message=None if passed else "monitor breach: " + "; ".join(failures),
        )
