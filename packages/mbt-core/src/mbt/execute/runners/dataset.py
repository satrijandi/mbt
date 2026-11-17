"""The dataset runner: materialize (or reuse) a dataset, then its checks and data tests."""

from pathlib import Path
from typing import Any

import pyarrow as pa

from mbt.artifacts.run_results import (
    NodeResult,
    TestResultEntry,
)
from mbt.dag.selector import evaluate_selector
from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import ConfigError
from mbt.execute.runners.context import (
    BuildContext,
    ExecutionContext,
    run_with_lifecycle,
    source_access,
)
from mbt.quality.checks import labeled_splits, run_checks
from mbt.quality.python_tests import PythonTestFile, run_python_tests
from mbt.state.lock import materialization_lock
from mbt_adapter_base import (
    DatasetSpec,
    ManifestNode,
)


class DatasetRunner:
    """Materialize (or reuse) a dataset, then run its checks and data tests.

    ``--phase materialize`` stops after the first; ``--phase checks`` runs only the
    second, against what a ``materialize`` left in the dataset cache (ADR-35).
    """

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def run(self, uid: str) -> NodeResult:
        node = self.ctx.manifest.nodes[uid]
        return run_with_lifecycle(self.ctx, uid, "dataset", lambda: self._run_inner(uid, node))

    def _run_inner(self, uid: str, node: ManifestNode) -> NodeResult:
        spec = DatasetSpec.model_validate(node.config)
        phase = self.ctx.phase
        if phase == "checks":
            # Builds nothing: a missing materialization is the error it always is.
            handle = self.ctx.dataset_handle(uid)
        else:
            handle = self._materialize(node, spec)
            self.ctx.store_dataset_handle(uid, handle)
        if phase == "materialize":
            return NodeResult(unique_id=uid, status="success")
        tests = self._run_quality(uid, node, spec, handle)
        failed = [t for t in tests if not t.passed]
        status = "test_failed" if failed else "success"
        message = (
            f"{len(failed)} check/test failure(s): "
            + "; ".join(f"{t.name}: {t.message}" for t in failed)
            if failed
            else None
        )
        return NodeResult(
            unique_id=uid,
            status=status,  # type: ignore[arg-type]
            tests=[TestResultEntry(**t.model_dump()) for t in tests],
            message=message,
        )

    def _materialize(self, node: ManifestNode, spec: DatasetSpec) -> Any:
        ctx = self.ctx
        output_dir = ctx.dataset_dir(node)

        def waiting() -> None:
            get_bus().emit(
                LogMessage(
                    unique_id=node.unique_id,
                    message=(
                        f"another process is materializing {node.name} ({output_dir.name}); "
                        "waiting for it"
                    ),
                )
            )

        cached = ctx.cached_dataset(node)
        if cached is None:
            # The dataset cache can be shared between invocations (ADR-32), so
            # a build is one critical section per key: the second builder
            # waits, then finds the first one's _SUCCESS instead of emptying
            # the directory under it. Probing first keeps a warm cache free of
            # lock files, and readable from a read-only mount.
            with materialization_lock(output_dir, on_wait=waiting):
                cached = ctx.cached_dataset(node)
                if cached is None:
                    return self._build(node, spec, output_dir)
        get_bus().emit(
            LogMessage(unique_id=node.unique_id, message=f"cache hit ({output_dir.name})")
        )
        return cached

    def _build(self, node: ManifestNode, spec: DatasetSpec, output_dir: Path) -> Any:
        ctx = self.ctx
        sample_fraction = ctx.sample_fraction()
        source_tables = {
            dep: ctx.manifest.sources[dep].config
            for dep in node.depends_on
            if dep.startswith("source.") and dep in ctx.manifest.sources
        }
        if not source_tables:
            raise ConfigError(
                f"dataset {node.name!r} has no source in the manifest",
                resource=node.unique_id,
            )
        if spec.source not in source_tables:
            raise ConfigError(
                f"dataset {node.name!r} source {spec.source!r} missing from the manifest",
                resource=node.unique_id,
                hint="recompile: the manifest and spec disagree",
            )
        windows = {
            split: (bounds[0], bounds[1])
            for split, bounds in node.resolved.get("windows", {}).items()
        }
        build_ctx = BuildContext(
            node=node,
            source=source_tables[spec.source],
            source_tables=source_tables,
            resolved_windows=windows,
            sample_fraction=sample_fraction,
            deep_snapshot=ctx.manifest.metadata.deep_snapshot,
            output_dir=output_dir,
            events=get_bus(),
            build_parallelism=ctx.threads,
        )
        return ctx.data_adapter.build_dataset(spec, build_ctx)

    def _run_quality(
        self, uid: str, node: ManifestNode, spec: DatasetSpec, handle: Any
    ) -> list[Any]:
        sources = source_access(self.ctx, node)
        results = list(run_checks(spec, handle, node.resolved, resource=uid, sources=sources))
        if self.ctx.command in ("build", "test"):
            results.extend(self._run_python_tests(uid, spec, handle))
        return results

    def _run_python_tests(self, uid: str, spec: DatasetSpec, handle: Any) -> list[Any]:
        bound = [tf for tf in self.ctx.python_tests if self._binds(tf, uid, spec)]
        if not bound:
            return []
        # Data tests are written against labelled rows; the after-test split may
        # hold immature ones with a null label, and may be empty (ADR-30).
        table = pa.concat_tables([handle.read(split) for split in labeled_splits(handle)])
        results = []
        for test_file in bound:
            only = set(spec.tests) & set(test_file.test_names) if spec.tests else None
            results.extend(run_python_tests(test_file, table, spec, only=only, resource=uid))
        return results

    def _binds(self, test_file: PythonTestFile, uid: str, spec: DatasetSpec) -> bool:
        if spec.tests and set(spec.tests) & set(test_file.test_names):
            return True
        if test_file.selector is None:
            return not spec.tests
        matched = evaluate_selector(test_file.selector, self.ctx.graph(), self.ctx.selectable())
        return uid in matched
