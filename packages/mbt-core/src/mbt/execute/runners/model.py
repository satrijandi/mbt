"""The model runner: train in a job, judge, register, finish; and ``mbt test``'s runner."""

from pathlib import Path
from typing import Any, Literal

from mbt.artifacts.run_results import (
    GateResult,
    MonitorResult,
    NodeResult,
    RegistrationResult,
)
from mbt.events import get_bus
from mbt.events.models import ArtifactRegistered, LogMessage
from mbt.events.node_log import combined_log
from mbt.exceptions import AdapterError, ConfigError, StateError
from mbt.execute.phases import ModelPhaseState, read_state, write_state
from mbt.execute.runners.champion_config import read_inference_config
from mbt.execute.runners.context import (
    ExecutionContext,
    evaluation_node_result,
    run_with_lifecycle,
)
from mbt.quality.gates import all_gates_passed, evaluate_gates
from mbt.quality.judgement import after_test_tags, judge
from mbt.quality.metrics import resolve_model_metrics
from mbt.runtime import (
    resolve_artifact_store_uri,
)
from mbt.utils import canonical_json
from mbt_adapter_base import (
    AdapterRef,
    ArtifactRef,
    JobResult,
    ManifestNode,
    MetricSpec,
    ModelSpec,
    ModelVersion,
    Stage,
    TrainingJob,
)
from mbt_adapter_base.capabilities import Capability, supports
from mbt_adapter_base.champion import (
    GATES_PASSED,
    AfterTestVerdict,
    ChampionRecord,
    has_inference_config,
    operating_points,
)


class ModelRunner:
    """Coordinator side of a model build (TSD §10.5): jobs compute, core compares."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        result = run_with_lifecycle(self.ctx, uid, "model", lambda: self._run_inner(uid, node))
        if self.ctx.phase is not None:
            self._add_to_elapsed(node, result)
        return result

    def _add_to_elapsed(self, node: ManifestNode, result: NodeResult) -> None:
        """Report the node's time so far, not only this phase's.

        A dashboard reads the last result a node pushed, and the last phase of a model
        is the registry write: its duration alone would say training took a second.
        The events still carry each phase's own time. A phase that left no state (a
        training that failed before it wrote one) has nothing to add to.
        """
        path = self.ctx.layout.phase_state(node.name)
        try:
            state = read_state(
                path, node, self.ctx.manifest.manifest_hash(), phase=self.ctx.phase or ""
            )
        except StateError:
            return
        state.elapsed_s += result.execution_time_s
        write_state(path, state)
        result.execution_time_s = state.elapsed_s

    # -- helpers -------------------------------------------------------------

    def _metric_specs(self, spec: ModelSpec, node: ManifestNode) -> list[MetricSpec]:
        from mbt.config.tasks import get_task_schema

        declared = {
            name: MetricSpec.model_validate(payload)
            for name, payload in self.ctx.manifest.metrics.items()
        }
        resolved, errors = resolve_model_metrics(
            spec,
            declared,
            get_task_schema(spec.task),
            has_hooks=node.hooks_path is not None,
        )
        if errors:
            raise ConfigError("; ".join(errors), resource=node.unique_id)
        return resolved

    def _champion(
        self, spec: ModelSpec, node: ManifestNode
    ) -> tuple[ModelVersion | None, Stage | None]:
        stages = {g.compare_to for g in spec.evaluation.gates if g.compare_to is not None}
        if not stages:
            return None, None
        stage = next(iter(stages))
        registry_name = spec.registration.name if spec.registration else spec.name
        champion = self.ctx.registry_adapter().get_champion(registry_name, stage)
        if champion is not None and champion.artifact is None:
            raise AdapterError(
                f"champion {registry_name} v{champion.version} in {stage.value!r} has no "
                "loadable artifact reference",
                resource=node.unique_id,
                hint="a champion that exists but cannot load is an error (ADR-10)",
            )
        return champion, stage

    def _champion_inference(self, champion: ModelVersion | None) -> dict[str, Any]:
        """The champion's own spec and fitted feature columns (ADR-28), so the
        champion gate scores it through what it was trained on, not through
        the challenger's feature selection (ADR-9). Empty for no champion, and
        for one registered before mbt exported an inference config - which the
        job then scores through the challenger's view, as it always did."""
        if champion is None or not has_inference_config(champion.tags):
            return {}
        document = read_inference_config(self.ctx, champion)
        resolved = document.get("resolved") or {}
        return {
            "champion_spec": document["spec"],
            "champion_feature_columns": resolved.get("feature_columns"),
        }

    def _assemble_job(
        self,
        node: ManifestNode,
        spec: ModelSpec,
        metric_specs: list[MetricSpec],
        champion: ModelVersion | None,
        *,
        mode: str = "train",
        artifact: ArtifactRef | None = None,
        champion_spec: dict[str, Any] | None = None,
        champion_feature_columns: list[str] | None = None,
        tracking_run_id: str | None = None,
    ) -> TrainingJob:
        ctx = self.ctx
        dataset_uid = next(d for d in node.depends_on if d.startswith("dataset."))
        dataset_node = ctx.manifest.nodes[dataset_uid]
        handle = ctx.dataset_handle(dataset_uid)
        tuning_cap = ctx.merged_vars.get("max_tuning_trials")
        meta = ctx.manifest.metadata
        return TrainingJob(
            mode=mode,  # type: ignore[arg-type]
            run_id=ctx.run_id,
            project_dir=str(ctx.project_dir),
            target_name=meta.target,
            project=meta.project_name,
            anchor=meta.anchor,
            node=node,
            dataset=handle.locator(),
            dataset_windows=dict(dataset_node.resolved),
            dataset_node=dataset_node,
            data=ctx.raw_adapter_ref("data"),
            tracking=ctx.raw_adapter_ref("tracking"),
            metric_specs=metric_specs,
            champion=champion.artifact if champion else None,
            artifact=artifact,
            # Set here rather than patched onto the returned job by the caller
            # (C-3): ``artifact`` already had a parameter this path bypassed in
            # favour of a model_copy, so there were two ways to set one field
            # and no way at all to set ``champion_spec``.
            champion_spec=champion_spec,
            champion_feature_columns=champion_feature_columns,
            tracking_run_id=tracking_run_id,
            tuning_engine=(
                # Engine name from the spec; ops config (sampler/pruner knobs)
                # from the target profile, so tuning knobs stay out of model
                # identity (same rationale as the pruner defaults).
                AdapterRef(
                    adapter=spec.tuning.engine,
                    config=dict(ctx.profiles.target.tuning.config)
                    if ctx.profiles.target.tuning is not None
                    else {},
                )
                if spec.tuning is not None
                else None
            ),
            tuning_cap=int(tuning_cap) if tuning_cap is not None else None,
            artifact_store=resolve_artifact_store_uri(
                ctx.profiles.target.artifact_store, ctx.project_dir
            ),
            required_env=list(ctx.profiles.required_env),
            tracking_meta={
                "mbt.run_id": ctx.run_id,
                # One run per model per invocation, named for the invocation
                # (ADR-28): a re-train tomorrow no longer produces a second
                # run with today's name. Composed here, beside the experiment
                # name, so the policy is not restated in every tracker.
                "mbt.run_name": f"{ctx.run_id}-{node.name}",
                "mbt.project": meta.project_name,
                "mbt.config_hash": node.config_hash,
                "mbt.input_hash": node.input_hash,
                "mbt.manifest_hash": ctx.manifest.manifest_hash(),
                "mbt.snapshot_id": dataset_node.snapshot_id or "",
                "mbt.git_commit": meta.git.commit or "",
            },
            vars=ctx.job_safe_vars(),
        )

    def _note_window_reuse(self, node: ManifestNode, uid: str) -> None:
        """Count how often this test window has arbitrated a gate (D-3).

        A held-out window selected against enough times stops being held out:
        the decisions are being made by the analyst reading pass/fail and
        adjusting. Counting is all this does - ADR-30's after-test window is
        the control, and it is already present.
        """
        from mbt.state.gate_log import record_evaluation, reuse_warning

        dataset_uid = next((d for d in node.depends_on if d.startswith("dataset.")), None)
        if dataset_uid is None:
            return
        dataset_node = self.ctx.manifest.nodes.get(dataset_uid)
        windows = (dataset_node.resolved.get("windows") or {}) if dataset_node else {}
        test = windows.get("test")
        window = (str(test[0]), str(test[1])) if test and len(test) == 2 else None
        use = record_evaluation(self.ctx.layout.gate_history, dataset_uid, window)
        if use is not None and use.overused:
            get_bus().emit(LogMessage(level="warn", unique_id=uid, message=reuse_warning(use)))

    # -- public seams for the pre-deploy check (C-3) --------------------------

    def check_job(
        self,
        node: ManifestNode,
        spec: ModelSpec,
        *,
        artifact: ArtifactRef,
        champion_spec: dict[str, Any],
        champion_feature_columns: list[str] | None,
        tracking_run_id: str | None,
    ) -> tuple[TrainingJob, list[MetricSpec]]:
        """The job an after-test check runs, and the metric specs it judges with.

        ``oot_check`` used to reach through four private methods
        (``_metric_specs``, ``_assemble_job``, ``_upload_log``,
        ``_gate_results``) and then patch the returned job (C-3). This is the
        seam that path actually needs.
        """
        metric_specs = self._metric_specs(spec, node)
        job = self._assemble_job(
            node,
            spec,
            metric_specs,
            None,
            mode="oot_check",
            artifact=artifact,
            champion_spec=champion_spec,
            champion_feature_columns=champion_feature_columns,
            tracking_run_id=tracking_run_id,
        )
        return job, metric_specs

    def after_test_gates(
        self,
        spec: ModelSpec,
        node: ManifestNode,
        job_result: Any,
        metric_specs: list[MetricSpec],
    ) -> list[GateResult]:
        """Evaluate ONLY the after-test gates (ADR-30)."""
        return self._gate_results(spec, node, job_result, None, metric_specs, out_of_time="only")

    def upload_run_log(self, node: ManifestNode, tracking_run_id: str | None, *, name: str) -> None:
        """Attach this run's node log to the tracking run."""
        self._upload_log(node, tracking_run_id, name=name)

    def _gate_results(
        self,
        spec: ModelSpec,
        node: ManifestNode,
        job_result: Any,
        champion_version: str | None,
        metric_specs: list[MetricSpec],
        *,
        evaluated_is_champion: bool = False,
        out_of_time: Literal["include", "only", "skip"] = "include",
    ) -> list[GateResult]:
        adapter = self.ctx.registry.training(spec.adapter)
        return evaluate_gates(
            spec.evaluation.gates,
            resource=node.unique_id,
            challenger=job_result.metrics,
            champion=job_result.champion_metrics,
            champion_version=champion_version,
            metric_specs=metric_specs,
            determinism=adapter.determinism,
            champion_delta_bounds=job_result.champion_delta_bounds,
            backtest_metrics=job_result.backtest_metrics,
            evaluated_is_champion=evaluated_is_champion,
            report=job_result.report,
            out_of_time=out_of_time,
        )

    def _register(
        self,
        spec: ModelSpec,
        node: ManifestNode,
        state: ModelPhaseState,
        persist: Path | None = None,
    ) -> RegistrationResult | None:
        """Register the trained artifact and set its stage.

        With ``persist`` (the ``register`` phase), the version is written to the phase
        state the moment the registry has it, so a retry after a failure in the stage
        transition finishes that version instead of registering the artifact again.
        """
        job_result = state.job_result
        if spec.registration is None or job_result.artifact is None:
            return None
        ctx = self.ctx
        registry_adapter = ctx.registry_adapter()
        dataset_uid = next(d for d in node.depends_on if d.startswith("dataset."))
        # One model owns the champion contract (A-5); core no longer spells
        # 53 keys by hand. The baseline is what scoring runs compare against
        # (ADR-21) and the inference config is the spec they reconstruct this
        # champion from (ADR-28), pinned the same way so the two travel together.
        record = ChampionRecord(
            artifact=job_result.artifact,
            config_hash=node.config_hash,
            input_hash=node.input_hash,
            manifest_hash=ctx.manifest.manifest_hash(),
            snapshot_id=ctx.manifest.nodes[dataset_uid].snapshot_id or "",
            git_commit=ctx.manifest.metadata.git.commit or "",
            tracking_run_id=job_result.tracking_run_id or "",
            hooks_hash=node.hooks_hash or "",  # scoring-time parity (ADR-20)
            gates_passed=True,
            baseline=job_result.baseline,
            inference_config=job_result.inference_config,
            report_uri=(job_result.report.report_uri if job_result.report else None) or None,
            # The after-test verdict at training time (ADR-30); a pre-deploy
            # check replaces it with a fresher one.
            after_test=(
                AfterTestVerdict(
                    passed=state.verdict, anchor=ctx.manifest.metadata.anchor, source="build"
                )
                if state.verdict is not None
                else None
            ),
            # Deployable cutoffs (R2-5), so a scoring pipeline can default its
            # decision rule from the model instead of a hand-copied constant.
            operating_points=(
                operating_points(job_result.metrics.metrics) if job_result.metrics else {}
            ),
        )
        if state.registered_version is not None:
            version = ModelVersion(name=spec.registration.name, version=state.registered_version)
        else:
            version = registry_adapter.register(
                job_result.artifact, spec.registration.name, record.pack()
            )
            if persist is not None:
                state.registered_version = version.version
                write_state(persist, state)
        registry_adapter.transition(version, spec.registration.stage_on_pass)
        get_bus().emit(
            ArtifactRegistered(
                unique_id=node.unique_id,
                registry=ctx.profiles.target.registry.adapter,
                name=spec.registration.name,
                version=version.version,
                stage=spec.registration.stage_on_pass.value,
            )
        )
        return RegistrationResult(
            registry=ctx.profiles.target.registry.adapter,
            name=spec.registration.name,
            version=version.version,
            stage=spec.registration.stage_on_pass.value,
        )

    def _attach_tracking_tags(
        self,
        job_result: Any,
        gates: list[GateResult],
        registration: RegistrationResult | None,
        stability: list[MonitorResult] | None = None,
        verdict: str | None = None,
    ) -> None:
        if job_result.tracking_run_id is None:
            return
        tags = {
            GATES_PASSED: str(all_gates_passed(gates)).lower(),
            "mbt.gates": canonical_json([g.model_dump(mode="json") for g in gates]),
        }
        if stability:
            tags["mbt.stability"] = canonical_json([m.model_dump(mode="json") for m in stability])
        if verdict is not None:
            tags.update(after_test_tags(verdict, self.ctx.manifest.metadata.anchor, "build"))
        if registration is not None:
            tags["mbt.registered_version"] = registration.version
            tags["mbt.registered_stage"] = registration.stage
        try:
            tracking = self.ctx.tracking()
            run = tracking.resume(job_result.tracking_run_id)
            tracking.log(run, tags=tags)
        except Exception as exc:
            get_bus().emit(
                LogMessage(level="warn", message=f"could not attach tracking tags: {exc}")
            )

    def _upload_log(
        self, node: ManifestNode, tracking_run_id: str | None, *, name: str = "train.log"
    ) -> None:
        """Put the node's log - its dataset's section first - on the run (ADR-30).

        Everything up to registration and the gate tags is in it; the node's
        final status line is emitted after this returns, so it is not.
        """
        logs = self.ctx.node_logs
        if logs is None or tracking_run_id is None:
            return
        import tempfile

        datasets = [dep for dep in node.depends_on if dep.startswith("dataset.")]
        text = combined_log(
            logs,
            [*datasets, node.unique_id],
            header=[
                f"mbt {self.ctx.command} - {node.unique_id}",
                f"run {self.ctx.run_id} on target {self.ctx.manifest.metadata.target}, "
                f"anchor {self.ctx.manifest.metadata.anchor}",
            ],
        )
        try:
            tracking = self.ctx.tracking()
            run = tracking.resume(tracking_run_id)
            with tempfile.TemporaryDirectory(prefix="mbt-log-") as staging:
                path = Path(staging) / name
                path.write_text(text, encoding="utf-8")
                if supports(tracking, Capability.LOG_DIRECTORY):
                    tracking.log_directory(run, Path(staging), "logs")
                elif supports(tracking, Capability.LOG_DOCUMENT):
                    tracking.log_document(run, path)
        except Exception as exc:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=node.unique_id,
                    message=f"could not upload the run log to the tracker: {exc}",
                )
            )

    # -- main path -------------------------------------------------------------

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        """Train, judge, register: the whole model, or the one phase asked for (ADR-35).

        Run whole, the three steps share ``state`` in memory and nothing is written.
        Run as phases, ``state`` is the file the steps hand to one another, and each
        step does what it always did, in the same order.
        """
        phase = self.ctx.phase
        spec = ModelSpec.model_validate(node.config)
        path = self.ctx.layout.phase_state(node.name)
        if phase in (None, "train"):
            trained = self._train(uid, node, spec)
            if isinstance(trained, NodeResult):
                return trained
            state = trained
            if phase == "train":
                write_state(path, state)
                return self._node_result(uid, state)
        else:
            state = read_state(path, node, self.ctx.manifest.manifest_hash(), phase=phase)

        if phase in (None, "gate"):
            self._judge(uid, node, spec, state)
            if phase == "gate":
                if not state.passed:
                    # A model that fails its gate goes no further, so this is its end.
                    self._finish(node, state)
                write_state(path, state)
                return self._node_result(uid, state)
        elif state.passed is None:
            raise StateError(
                f"model {node.name!r} has not been judged: its gates have not run",
                resource=uid,
                hint="run 'mbt run --phase gate' for this model before 'register'",
            )
        elif not state.passed:
            raise StateError(
                f"model {node.name!r} failed its gates, so there is nothing to register",
                resource=uid,
                hint=state.failure_summary,
            )

        if state.passed and state.registration is None:
            state.registration = self._register(
                spec, node, state, persist=path if phase == "register" else None
            )
        self._finish(node, state)
        if phase == "register":
            write_state(path, state)
        return self._node_result(uid, state)

    def _train(self, uid: str, node: ManifestNode, spec: ModelSpec) -> ModelPhaseState | NodeResult:
        """Run the training job; its result, or the error NodeResult it ended in."""
        metric_specs = self._metric_specs(spec, node)
        champion, _stage = self._champion(spec, node)

        job = self._assemble_job(
            node, spec, metric_specs, champion, **self._champion_inference(champion)
        )
        job_result = self.ctx.run_job(job)

        if job_result.status == "error" or job_result.metrics is None:
            self._upload_log(node, job_result.tracking_run_id)
            return NodeResult(
                unique_id=uid,
                status="error",
                message=job_result.error or "job returned no metrics",
            )
        return ModelPhaseState(
            unique_id=uid,
            manifest_hash=self.ctx.manifest.manifest_hash(),
            job_result=job_result,
            champion_version=champion.version if champion else None,
        )

    def _judge(self, uid: str, node: ManifestNode, spec: ModelSpec, state: ModelPhaseState) -> None:
        """Judge the job result against the gates and record the judgement on ``state``."""
        metric_specs = self._metric_specs(spec, node)
        gates = self._gate_results(
            spec, node, state.job_result, state.champion_version, metric_specs
        )
        # One owner for "judge this job result" (C-1).
        outcome = judge(spec, gates, state.job_result.report, resource=uid)
        self._note_window_reuse(node, uid)
        state.gates = gates
        state.stability = outcome.stability.results
        state.verdict = outcome.verdict
        state.passed = outcome.passed
        state.failure_summary = outcome.failure_summary

    def _finish(self, node: ManifestNode, state: ModelPhaseState) -> None:
        """What ends a model's run, passed or not: the tracking tags and the run log."""
        job_result = state.job_result
        self._attach_tracking_tags(
            job_result, state.gates or [], state.registration, state.stability, state.verdict
        )
        self._upload_log(node, job_result.tracking_run_id)

    @staticmethod
    def _node_result(uid: str, state: ModelPhaseState) -> NodeResult:
        job_result = state.job_result
        assert job_result.metrics is not None  # _train returns an error result without them
        return NodeResult(
            unique_id=uid,
            # A model that has not been judged yet (after `train`) has failed nothing.
            status="gate_failed" if state.passed is False else "success",
            metrics=dict(job_result.metrics.metrics),
            slices=dict(job_result.metrics.slices),
            gates=state.gates or [],
            artifact=job_result.artifact,
            registration=state.registration,
            tracking_run_id=job_result.tracking_run_id,
            resolved_auto=dict(job_result.resolved_auto),
            feature_importance=dict(job_result.feature_importance),
            partial_dependence=dict(job_result.partial_dependence),
            backtest_metrics=dict(job_result.backtest_metrics),
            backtest_std=dict(job_result.backtest_std),
            monitors=state.stability,
            message=state.failure_summary,
        )

    # -- re-evaluation of registered artifacts (FR-RUN-07, TSD §11.3) -----------

    def evaluate_artifact(
        self,
        node: ManifestNode,
        spec: ModelSpec,
        artifact: ArtifactRef,
        *,
        apply_gates: bool = True,
    ) -> tuple[JobResult, list[GateResult]]:
        """Run an evaluate-mode job for a registered artifact; never trains.

        The one flow behind both ``mbt test`` on models and ``mbt evaluate``,
        so the two commands cannot drift: fresh data, the stored artifact,
        gate logic against the current champion when requested.
        """
        metric_specs = self._metric_specs(spec, node)
        champion, _stage = self._champion(spec, node) if apply_gates else (None, None)
        # Re-evaluating the champion itself: loading it a second time as its own
        # comparator would only produce a zero delta (see evaluate_gates).
        is_champion = (
            champion is not None
            and champion.artifact is not None
            and champion.artifact.uri == artifact.uri
        )
        compared = None if is_champion else champion
        job = self._assemble_job(
            node,
            spec,
            metric_specs,
            compared,
            mode="evaluate",
            artifact=artifact,
            **self._champion_inference(compared),
        )
        job_result = self.ctx.run_job(job)
        gates: list[GateResult] = []
        if (
            apply_gates
            and job_result.status != "error"
            and job_result.metrics is not None
            and spec.evaluation.gates
        ):
            gates = self._gate_results(
                spec,
                node,
                job_result,
                champion.version if champion else None,
                metric_specs,
                evaluated_is_champion=is_champion,
                # A re-evaluation builds no after-test report; those gates are
                # judged at build time and by `mbt evaluate --out-of-time`.
                out_of_time="skip",
            )
        return job_result, gates


class ModelTestRunner:
    """``mbt test`` for models: re-evaluate the latest registered version;
    training is never a side effect (TSD §11.3)."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx
        self._model_runner = ModelRunner(ctx)

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "model", lambda: self._run_inner(uid, node))

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        ctx = self.ctx
        spec = ModelSpec.model_validate(node.config)
        if not spec.evaluation.gates:
            return NodeResult(unique_id=uid, status="skipped", message="model declares no gates")
        registry_name = spec.registration.name if spec.registration else spec.name
        stage = spec.registration.stage_on_pass if spec.registration else Stage.STAGING
        version = ctx.registry_adapter().get_champion(registry_name, stage)
        if version is None or version.artifact is None:
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=uid,
                    message=(
                        f"no registered version of {registry_name!r} in {stage.value!r}; "
                        "skipping model tests (mbt test never trains)"
                    ),
                )
            )
            return NodeResult(unique_id=uid, status="skipped", message="no registered version")

        job_result, gates = self._model_runner.evaluate_artifact(node, spec, version.artifact)
        return evaluation_node_result(
            uid,
            job_result,
            gates,
            pass_status="success",
            fail_status="test_failed",
            include_feature_importance=True,
            include_message=True,
        )
