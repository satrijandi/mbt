"""The training job's report step (ADR-30).

Scores each split exactly once - train (in-sample), test as the metrics saw
it, and the after-test window - aligns the scores with the raw key and time
columns, builds the report, writes it, and puts it in the artifact store and
on the tracking run. The job imports this lazily, after it has trained.
"""

import contextlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from mbt.events import get_bus
from mbt.events.bus import HookEventSink
from mbt.events.models import LogMessage
from mbt.exceptions import ConfigError
from mbt.execute.handles import SplitRouter, TrainingSplitView, TransformedDatasetHandle
from mbt.execute.job_runtime import JobRuntime
from mbt.execute.seeds import permutation_importance_seed
from mbt.reporting.builder import ReportData, ReportInputs, ScoredSplit, TableKey
from mbt.reporting.writer import ReportMeta, write_report
from mbt_adapter_base import (
    OUT_OF_TIME_SPLIT,
    DatasetSpec,
    ModelSpec,
    ReportSummary,
    TaskType,
)
from mbt_adapter_base.capabilities import Capability, capabilities_of

if TYPE_CHECKING:
    import numpy as np

#: Rows the permutation fallback scores per feature, and the most features it
#: will shuffle - each one is a full predict call.
PERMUTATION_ROWS = 5000
PERMUTATION_MAX_FEATURES = 100

#: Shuffles per feature, averaged (D-5). A single shuffle is a high-variance
#: estimate of a drop, and normalising it to a fraction made the result read as
#: settled when two runs at different seeds could reorder mid-table features.
#: sklearn's ``permutation_importance`` defaults to 5 for the same reason.
PERMUTATION_REPEATS = 5

Stager = Callable[[Any, ModelSpec], Any]


# -- inputs ------------------------------------------------------------------------------------


def dataset_spec(runtime: JobRuntime) -> DatasetSpec | None:
    node = runtime.job.dataset_node
    if node is None:
        return None
    return DatasetSpec.model_validate(node.config)


def job_anchor(runtime: JobRuntime) -> datetime:
    """The manifest anchor (ADR-12); maturity is measured against it."""
    raw = str(runtime.job.anchor or "")
    if not raw:
        return datetime.now(tz=UTC).replace(microsecond=0)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def report_view(runtime: JobRuntime) -> TrainingSplitView:
    """The training view with the after-test split visible, test narrowed
    exactly as the metrics saw it."""
    test_window = runtime.job.node.resolved.get("test_window")
    return TrainingSplitView(
        runtime.materialization,
        time_column=getattr(runtime.materialization, "time_column", None),
        test_window=(str(test_window[0]), str(test_window[1])) if test_window else None,
        hide_out_of_time=False,
    )


def _labels(table: pa.Table, target: str) -> "np.ndarray":
    import numpy as np
    import pyarrow.compute as pc

    column = table.column(target)
    if pa.types.is_boolean(column.type):
        column = pc.cast(column, pa.int8())
    values = pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=float)


def _scored_splits(
    runtime: JobRuntime,
    model: Any,
    *,
    stage: Stager,
    include_train: bool = True,
) -> dict[str, ScoredSplit]:
    """Predict each report split once and align it with its raw rows.

    Keys and times come from the raw split by position, the rule scoring
    already enforces (ADR-20): when the report needs them - row-level
    predictions, or an after-test window to cut into periods - a hook that
    changes the row count is an error rather than a silent misalignment.
    """
    from mbt_adapter_base.reporting import time_values

    spec = runtime.spec
    view = report_view(runtime)
    dataset = dataset_spec(runtime)
    keys = dataset.sample_key_columns if dataset is not None else []
    time_column = view.time_column
    report = spec.evaluation.effective_report
    has_after = OUT_OF_TIME_SPLIT in view.splits() and _row_count(view, OUT_OF_TIME_SPLIT) > 0
    need_alignment = report.predictions.enabled or has_after

    routes: dict[str, Any] = {"test": runtime.handle}
    if include_train and "train" in view.splits():
        routes["train"] = runtime.handle
    if has_after:
        after: Any = TransformedDatasetHandle(
            view,
            spec,
            runtime.hooks,
            lambda split: _hook_context(runtime, split),
            time_column,
            pinned_features=runtime.transformed.feature_columns(),
        )
        if getattr(runtime.adapter, "data_access", "arrow") == "path":
            after = stage(_Only(after, OUT_OF_TIME_SPLIT), spec)
        routes[OUT_OF_TIME_SPLIT] = after
    router = SplitRouter(routes)

    out: dict[str, ScoredSplit] = {}
    for split in (*(s for s in ("train", "test") if s in routes), OUT_OF_TIME_SPLIT):
        if split == OUT_OF_TIME_SPLIT and not has_after:
            continue
        raw = view.read(split)
        predictions: pa.Table = runtime.adapter.predict(model, router, split)
        passthrough_columns = [c for c in dict.fromkeys([*keys, time_column or ""]) if c]
        aligned = predictions.num_rows == raw.num_rows
        if not aligned and need_alignment:
            raise ConfigError(
                f"hooks changed the {split} split's row count ({raw.num_rows} rows in, "
                f"{predictions.num_rows} predictions out), so the training report cannot "
                "line predictions up with their keys and dates",
                resource=runtime.job.node.unique_id,
                hint="transform_features must be row-stable (no filtering or reordering), "
                "as scoring already requires (ADR-20)",
            )
        present = [c for c in passthrough_columns if c in raw.column_names]
        out[split] = ScoredSplit(
            name=split,
            features=predictions.drop_columns(["prediction"]),
            passthrough=raw.select(present) if aligned else pa.table({}),
            key_columns=[c for c in keys if c in present],
            labels=_labels(predictions, spec.target),
            scores=predictions.column("prediction").to_numpy(zero_copy_only=False),
            times=time_values(raw.column(time_column))
            if aligned and time_column and time_column in raw.column_names
            else None,
        )
    return out


class _Only:
    """One split of a handle, so path staging writes just that file."""

    def __init__(self, handle: Any, split: str) -> None:
        self._handle = handle
        self._split = split

    @property
    def snapshot_id(self) -> str:
        return str(self._handle.snapshot_id)

    def splits(self) -> set[str]:
        return {self._split}

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        table: pa.Table = self._handle.read(split, columns)
        return table


def _row_count(view: TrainingSplitView, split: str) -> int:
    return int(view.profile().n_rows.get(split, 0))


def _hook_context(runtime: JobRuntime, split: str) -> Any:
    from mbt_adapter_base import (
        HookContext,
    )

    return HookContext(
        spec=runtime.spec,
        profile=runtime.base_profile,
        split=split,
        logger=HookEventSink(get_bus()),
    )


def report_inputs(
    runtime: JobRuntime, spec: ModelSpec, importance: dict[str, float]
) -> ReportInputs:
    dataset = dataset_spec(runtime)
    windows = dict(runtime.job.dataset_windows.get("windows") or {})
    after = windows.get(OUT_OF_TIME_SPLIT)
    return ReportInputs(
        after_window=(str(after[0]), str(after[1])) if after else None,
        model_name=spec.name,
        binary=spec.task == TaskType.BINARY_CLASSIFICATION,
        report=spec.evaluation.effective_report,
        feature_columns=runtime.transformed.feature_columns(),
        importance=importance,
        metric_specs=list(runtime.builtin_specs),
        anchor=job_anchor(runtime),
        horizon=dataset.label.horizon if dataset is not None else None,
        stability=spec.evaluation.stability,
        gate_periods={
            g.effective_period for g in spec.evaluation.gates if g.source == "out_of_time"
        },
    )


# -- importance --------------------------------------------------------------------------------


def permutation_importance(runtime: JobRuntime, model: Any, test: ScoredSplit) -> dict[str, float]:
    """Model-agnostic importance for adapters that report none.

    Shuffles one feature at a time on a seeded sample of the test split and
    measures how much the primary metric worsens (ROC AUC for a classifier,
    RMSE for regression). Negative drops clip to zero. Capped because every
    feature costs a predict call.
    """
    import numpy as np

    from mbt_adapter_base import (
        MetricSpec,
    )
    from mbt_adapter_base.datasets import InMemoryDatasetHandle
    from mbt_adapter_base.metrics import compute_metric

    features = [c for c in runtime.transformed.feature_columns() if c in test.features.column_names]
    bus = get_bus()
    uid = runtime.job.node.unique_id
    if len(features) > PERMUTATION_MAX_FEATURES:
        bus.emit(
            LogMessage(
                level="warn",
                unique_id=uid,
                message=(
                    f"the {runtime.spec.adapter!r} adapter reports no feature importance and "
                    f"the model has {len(features)} features (more than "
                    f"{PERMUTATION_MAX_FEATURES}); skipping the permutation fallback"
                ),
            )
        )
        return {}
    labelled = ~np.isnan(test.labels)
    rng = np.random.default_rng(permutation_importance_seed(runtime.spec.seed))
    rows = np.flatnonzero(labelled)
    if rows.size > PERMUTATION_ROWS:
        rows = np.sort(rng.choice(rows, size=PERMUTATION_ROWS, replace=False))
    sample = test.features.take(pa.array(rows))
    n_rows = sample.num_rows
    labels = test.labels[rows]
    binary = runtime.spec.task == TaskType.BINARY_CLASSIFICATION
    if binary and np.unique(labels).size < 2:
        return {}
    metric = MetricSpec(name="roc_auc" if binary else "rmse")
    sign = 1.0 if binary else -1.0

    def score(table: pa.Table) -> float:
        handle = InMemoryDatasetHandle({"permutation": table}, label_column=runtime.spec.target)
        predicted = runtime.adapter.predict(model, handle, "permutation").column("prediction")
        return sign * float(
            compute_metric(metric, labels, predicted.to_numpy(zero_copy_only=False))
        )

    reference = score(sample)
    drops: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for name in features:
        index = sample.column_names.index(name)
        # Repeats, not a single shuffle (D-5). One shuffle is a high-variance
        # estimate, and normalising it to a tidy fraction made the output read
        # as more settled than it was - two runs at different seeds could
        # reorder mid-table features. sklearn's equivalent defaults to 5
        # repeats for exactly this reason. The cost is linear in repeats and
        # this path only runs for adapters that report no native importance.
        measured = [
            max(
                0.0,
                reference
                - score(
                    sample.set_column(
                        index, name, sample.column(name).take(pa.array(rng.permutation(n_rows)))
                    )
                ),
            )
            for _ in range(PERMUTATION_REPEATS)
        ]
        drops[name] = float(np.mean(measured))
        spreads[name] = float(np.std(measured))
    total = sum(drops.values())
    noisiest = max(spreads.items(), key=lambda kv: kv[1], default=("", 0.0))
    bus.emit(
        LogMessage(
            unique_id=uid,
            message=(
                f"feature importance by permutation on {sample.num_rows} test rows "
                f"({len(features)} features x {PERMUTATION_REPEATS} shuffles; the adapter "
                f"reports none; widest spread across shuffles: {noisiest[0] or 'n/a'} "
                f"+/-{noisiest[1]:.4f})"
            ),
        )
    )
    if total == 0:
        return dict.fromkeys(drops, 0.0)
    return {name: value / total for name, value in drops.items()}


# -- publishing --------------------------------------------------------------------------------


def dataset_record(runtime: JobRuntime) -> dict[str, Any] | None:
    """The dataset side of what the model was trained on (ADR-30): enough for a
    serving system to rebuild the input, and for the pre-deploy check to pin
    the test window the model was judged on."""
    node = runtime.job.dataset_node
    dataset = dataset_spec(runtime)
    if node is None or dataset is None:
        return None
    windows = {k: list(v) for k, v in dict(node.resolved.get("windows") or {}).items()}
    metadata = dict(getattr(runtime.materialization, "metadata", {}) or {})
    return {
        "unique_id": node.unique_id,
        "name": node.name,
        "config_hash": node.config_hash,
        "snapshot_id": node.snapshot_id,
        "source": dataset.source,
        "sample_key": dataset.sample_key_columns,
        "label": {"column": dataset.label.column, "horizon": dataset.label.horizon},
        "time_column": dataset.split.time_column,
        "split_strategy": dataset.split.strategy.value,
        "filters": list(dataset.filters),
        "windows": windows,
        "test_window": runtime.job.node.resolved.get("test_window"),
        "anchor": str(runtime.job.anchor or ""),
        "sample_fraction": metadata.get("sample_fraction"),
        "row_counts": metadata.get("row_counts") or {},
    }


def _report_meta(
    runtime: JobRuntime, spec: ModelSpec, splits: dict[str, ScoredSplit], *, kind: str
) -> ReportMeta:
    record = dataset_record(runtime) or {}
    windows = dict(record.get("windows") or {})
    if record.get("test_window"):
        windows["test"] = list(record["test_window"])
    meta = runtime.job.tracking_meta
    return ReportMeta(
        model=spec.name,
        run=meta.get("mbt.run_name") or runtime.job.run_id,
        anchor=str(runtime.job.anchor or ""),
        binary=spec.task == TaskType.BINARY_CLASSIFICATION,
        kind="training" if kind == "training" else "pre-deploy check",
        dataset=record.get("name"),
        windows={k: v for k, v in windows.items() if k in splits or k == "validation"},
        row_counts={name: split.n_rows for name, split in splits.items()},
    )


def build_report_data(
    runtime: JobRuntime,
    spec: ModelSpec,
    importance: dict[str, float],
    scored: dict[str, ScoredSplit],
) -> ReportData:
    """The report's numbers, any engine's drift reports, and its warnings logged."""
    from mbt.reporting.builder import build_report

    data = build_report(report_inputs(runtime, spec, importance), scored)
    render_drift(runtime, spec, data, scored)
    for message in data.summary.warnings:
        get_bus().emit(
            LogMessage(unique_id=runtime.job.node.unique_id, message=f"report: {message}")
        )
    return data


def render_drift(
    runtime: JobRuntime, spec: ModelSpec, data: ReportData, scored: dict[str, ScoredSplit]
) -> None:
    """A report engine's drift report per after-test cell (ADR-30).

    The whole window first, then the newest months up to
    ``max_html_reports``, each over the ``feature_top_n`` most important
    features and the score. Display only - mbt's own statistics gate - so an
    engine that fails is a warning on the page, never a failed build.
    """
    import atexit
    import shutil

    import numpy as np

    from mbt.adapters.registry import get_registry
    from mbt_adapter_base.reporting import cell_label, period_codes

    settings = spec.evaluation.effective_report.stability
    after = scored.get(OUT_OF_TIME_SPLIT)
    if settings.engine == "native" or after is None or after.times is None or not after.n_rows:
        return
    try:
        engine = get_registry().component("reporting", settings.engine, {})
    except Exception as exc:
        data.summary.warnings.append(f"{settings.engine} drift report skipped: {exc}")
        return
    test = scored["test"]
    # the model's own features only (the scored tables still carry the label),
    # most important first
    features = runtime.transformed.feature_columns()
    ranked = [str(row["feature"]) for row in data.tables.get("feature_importance", [])]
    ordered = dict.fromkeys([*(c for c in ranked if c in features), *features])
    present = set(after.features.column_names) & set(test.features.column_names)
    columns = [c for c in ordered if c in present][: settings.feature_top_n]

    cells: list[tuple[str, np.ndarray]] = [("window", np.ones(after.n_rows, dtype=bool))]
    codes = period_codes(after.times, "month")
    months = [int(code) for code in np.unique(codes.cells[codes.valid])]
    kept = months[-(settings.max_html_reports - 1) :] if settings.max_html_reports > 1 else []
    if len(kept) < len(months):
        data.summary.warnings.append(
            f"{settings.engine} drift reports cover the newest {len(kept)} of "
            f"{len(months)} after-test months (max_html_reports: {settings.max_html_reports})"
        )
    cells += [(cell_label(code, "month"), codes.valid & (codes.cells == code)) for code in kept]

    def frame(split: ScoredSplit, mask: np.ndarray) -> pa.Table:
        table = split.features.select(columns).filter(pa.array(mask))
        return table.append_column("prediction", pa.array(split.scores[mask], pa.float64()))

    reference = frame(test, np.ones(test.n_rows, dtype=bool))
    staging = Path(tempfile.mkdtemp(prefix="mbt-drift-"))
    # read back when the report is written; the job process is short-lived
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    summary_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    for key, mask in cells:
        out = staging / f"{key}.html"
        try:
            found = engine.drift_report(
                reference, frame(after, mask), out, title=f"{spec.name}: {key} against test"
            )
        except Exception as exc:
            data.summary.warnings.append(f"{settings.engine} drift report for {key} failed: {exc}")
            continue
        relative = f"stability/{settings.engine}/{key}.html"
        data.files[relative] = out
        summary_rows.append(
            {
                "cell": key,
                "n_rows": int(mask.sum()),
                "reference_rows": test.n_rows,
                "drifted_share": found.drifted_share,
                "file": relative,
            }
        )
        column_rows.extend({"cell": key, **c.model_dump()} for c in found.columns)
    data.engine = settings.engine
    data.tables[TableKey.ENGINE_DRIFT_SUMMARY] = summary_rows
    data.tables[TableKey.ENGINE_DRIFT] = column_rows


def _publish_report(
    runtime: JobRuntime,
    data: ReportData,
    meta: ReportMeta,
    *,
    tracking: Any,
    run_handle: Any,
    prefix: str,
    artifact_path: str | None = None,
) -> ReportSummary:
    """Write the report and put it next to the model and on the run.

    The artifact store copy is what survives a tracker outage and what
    ``mbt clean`` keeps with the model; the tracker upload is best effort,
    like every other document mbt logs. ``artifact_path`` places the report
    on the run when it differs from the store ``prefix``.
    """
    with tempfile.TemporaryDirectory(prefix="mbt-report-") as staging:
        root = Path(staging)
        documents = write_report(data, meta, root)
        page = runtime.store.put_file(root / "report.html", f"{prefix}/report.html", "report")
        data.summary.report_uri = page.uri
        # Rewritten now that it can say where the page landed.
        (root / "summary.json").write_text(
            json.dumps(data.summary.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"
        )
        for relative in documents:
            if relative != "report.html":
                runtime.store.put_file(root / relative, f"{prefix}/{relative}", "report")
        if tracking is not None and run_handle is not None:
            _upload(runtime, tracking, run_handle, root, artifact_path or prefix, documents)
    return data.summary


def _upload(
    runtime: JobRuntime,
    tracking: Any,
    run_handle: Any,
    root: Path,
    prefix: str,
    documents: list[str],
) -> None:
    try:
        if hasattr(tracking, "log_directory"):
            tracking.log_directory(run_handle, root, prefix)
        elif hasattr(tracking, "log_document"):
            for name in ("report.html", "summary.json"):
                if name in documents:
                    tracking.log_document(run_handle, root / name)
    except Exception as exc:  # a tracker hiccup must not fail training
        get_bus().emit(
            LogMessage(
                level="warn",
                unique_id=runtime.job.node.unique_id,
                message=f"could not upload the training report to the tracker: {exc}",
            )
        )


def log_config_documents(
    runtime: JobRuntime, tracking: Any, run_handle: Any, spec: ModelSpec
) -> None:
    """The resolved model config and the dataset config as JSON on the run -
    the full values behind any parameter a tracker truncated."""
    if not hasattr(tracking, "log_directory"):
        return
    with (
        tempfile.TemporaryDirectory(prefix="mbt-config-") as staging,
        contextlib.suppress(Exception),
    ):
        root = Path(staging)
        (root / "model_config.json").write_text(
            json.dumps(spec.model_dump(mode="json"), indent=1, sort_keys=True) + "\n"
        )
        record = dataset_record(runtime)
        if record is not None and runtime.job.dataset_node is not None:
            payload = {"spec": runtime.job.dataset_node.config, "resolved": record}
            (root / "dataset_config.json").write_text(
                json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
            )
        tracking.log_directory(run_handle, root, "config")


def report_metrics(summary: ReportSummary) -> dict[str, float]:
    """The report's headline numbers as tracking metrics.

    ``train.<m>`` beside the bare test metrics, ``oot.window.<m>`` and
    ``oot.month.<YYYY-MM>.<m>`` for the after-test cells, and the score and
    worst-feature PSI per stability cell. Week and day cells stay in the
    report's files: at one metric per day per month they would bury the
    run's metric list.
    """
    import math

    out: dict[str, float] = {}

    def put(key: str, value: float | None) -> None:
        if value is not None and math.isfinite(value):
            out[key] = float(value)

    for name, value in summary.split_metrics.get("train", {}).items():
        put(f"train.{name}", value)
    for cell in summary.periods:
        if cell.period == "window":
            prefix = "oot.window"
        elif cell.period == "month":
            prefix = f"oot.month.{cell.key}"
        else:
            continue
        for name, value in cell.metrics.items():
            put(f"{prefix}.{name}", value)
        put(f"{prefix}.n_labelled", float(cell.n_labelled))
    for stability in summary.stability:
        if stability.period == "window":
            prefix = "stability.window"
        elif stability.period == "month":
            prefix = f"stability.month.{stability.key}"
        else:
            continue
        put(f"{prefix}.score_psi", stability.score_psi)
        put(f"{prefix}.max_feature_psi", stability.max_feature_psi)
        put(f"{prefix}.features_over_fail", float(stability.features_over_fail))
    return out


# -- the one entry point (B-2) -----------------------------------------------------------------


@dataclass(frozen=True)
class ReportOutcome:
    """What one report run produced.

    ``summary`` is what most callers want; ``scored`` and ``importance`` are
    here because the training path reuses the test scores for the monitoring
    baseline and the importance for partial dependence, and recomputing either
    would mean scoring every split twice.
    """

    summary: ReportSummary
    scored: dict[str, ScoredSplit]
    importance: dict[str, float]
    data: ReportData


def feature_importance(
    runtime: JobRuntime, model: Any, scored: ScoredSplit | None = None
) -> dict[str, float]:
    """Per-feature importance for the model card (FR-DOCS-02).

    Prefers a data-grounded SHAP importance when the adapter declares it (the
    tree adapters), since split-gain is cardinality-biased; falls back to the
    adapter's model-intrinsic ``feature_importance``, and then to the
    model-agnostic permutation fallback.

    The three-way preference used to be two functions in two modules with the
    fallback written out at both of ``job.py``'s call sites (B-2).
    """
    adapter = runtime.adapter
    # Declared, not probed (B-1): a renamed method is a capability the adapter
    # stops declaring, rather than one that silently disappears.
    can = capabilities_of(adapter, runtime.spec)
    if Capability.SHAP_IMPORTANCE in can:
        return dict(adapter.shap_importance(model, runtime.handle, "test"))
    if Capability.FEATURE_IMPORTANCE in can:
        return dict(adapter.feature_importance(model))
    # The permutation fallback shuffles scored test rows, so it needs them;
    # a caller with no scored split (``mbt evaluate``) simply gets no table.
    if scored is None:
        return {}
    return permutation_importance(runtime, model, scored)


def produce_report(
    runtime: JobRuntime,
    spec: ModelSpec,
    model: Any,
    *,
    kind: str,
    stage: Stager,
    include_train: bool = True,
    tracking: Any = None,
    run_handle: Any = None,
    prefix: str = "report",
    artifact_path: str | None = None,
) -> ReportOutcome:
    """Score the report splits, build the report, publish it (ADR-30).

    The six-step sequence this replaces was an ordered protocol the CALLER had
    to restate, and it was restated twice in ``job.py`` - at the training path
    and at the pre-deploy check - differing only in ``include_train``, ``kind``
    and ``artifact_path``, which are the three parameters here (B-2).

    That is what made ``training_report.py`` a file split rather than a seam:
    the interface was as large as the implementation, so pasting the module
    back into ``job.py`` would have changed nothing a reader holds in their
    head. Now there is one call.
    """
    scored = _scored_splits(runtime, model, stage=stage, include_train=include_train)
    importance = feature_importance(runtime, model, scored["test"])
    data = build_report_data(runtime, spec, importance, scored)
    summary = _publish_report(
        runtime,
        data,
        _report_meta(runtime, spec, scored, kind=kind),
        tracking=tracking,
        run_handle=run_handle,
        prefix=prefix,
        artifact_path=artifact_path,
    )
    return ReportOutcome(summary=summary, scored=scored, importance=importance, data=data)
