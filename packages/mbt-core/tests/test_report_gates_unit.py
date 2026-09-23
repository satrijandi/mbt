"""Pure pieces behind the training report (ADR-30): gates, stability, verdicts,
node logs, flat parameters, the split router, and the model card."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
from exec_unit_helpers import recording_bus

from mbt.artifacts.run_results import GateResult, MonitorResult, NodeResult
from mbt.contracts import (
    DeterminismTier,
    GateSpec,
    MetricResults,
    MetricSpec,
    ModelSpec,
    MonitorStats,
    PeriodCell,
    ReportSummary,
    ShiftStat,
    StabilityCell,
    StabilitySpec,
)
from mbt.events.models import LogMessage
from mbt.events.node_log import INVOCATION_LOG, NodeLogSink, combined_log, node_scope
from mbt.execute.handles import SplitRouter
from mbt.quality.gates import evaluate_gates
from mbt.quality.judgement import after_test_tags, after_test_verdict, gate_failure_summary
from mbt.quality.monitors import StabilityOutcome, evaluate_stability
from mbt.reporting.flatten import dataset_params, flatten, model_params
from mbt_adapter_base.datasets import InMemoryDatasetHandle


def _cell(period: str, key: str, value: float, **extra: Any) -> PeriodCell:
    payload: dict[str, Any] = {
        "period": period,
        "key": key,
        "slot": "all",
        "start": "s",
        "end": "e",
        "n_rows": 50,
        "n_labelled": 50,
        "mature": True,
        "metrics": {"roc_auc": value, "logloss": value},
    }
    payload.update(extra)
    return PeriodCell(**payload)


def _report(*cells: PeriodCell, stability: list[StabilityCell] | None = None) -> ReportSummary:
    return ReportSummary(
        anchor="a",
        reference_rows=10,
        out_of_time_rows=10,
        periods=list(cells),
        stability=stability or [],
    )


def _gates(gates: list[dict[str, Any]], report: ReportSummary | None, **kwargs: Any) -> Any:
    return evaluate_gates(
        [GateSpec.model_validate(g) for g in gates],
        resource="model.demo.m",
        challenger=MetricResults(metrics={"roc_auc": 0.8, "logloss": 0.4}),
        champion=None,
        champion_version=None,
        metric_specs=[
            MetricSpec(name="roc_auc"),
            MetricSpec(name="logloss", greater_is_better=False),
        ],
        report=report,
        **kwargs,
    )


def test_after_test_gates_judge_the_worst_mature_cell() -> None:
    report = _report(
        _cell("month", "2026-05", 0.72),
        _cell("month", "2026-06", 0.61),
        _cell("month", "2026-07", 0.1, mature=False),
        _cell("month", "2026-08", 0.2, n_labelled=3),
        _cell("week_of_month", "2026-06/W1", 0.3),
    )
    (higher, lower) = _gates(
        [
            {"metric": "roc_auc", "threshold": 0.6, "source": "out_of_time", "min_rows": 10},
            {"metric": "logloss", "threshold": 0.7, "source": "out_of_time", "min_rows": 10},
        ],
        report,
    )
    assert (higher.cell, higher.actual, higher.passed) == ("2026-06", 0.61, True)
    assert higher.message == "worst month cell 2026-06 of 2 judged"
    assert (lower.cell, lower.actual, lower.passed) == ("2026-05", 0.72, False)


def test_after_test_gates_use_the_determinism_tolerance() -> None:
    report = _report(_cell("month", "2026-06", 0.59))
    (gate,) = _gates(
        [{"metric": "roc_auc", "threshold": 0.6, "source": "out_of_time", "min_rows": 1}],
        report,
        determinism=DeterminismTier(kind="tolerance", tolerances={"roc_auc": 0.02}),
    )
    assert gate.passed


def test_an_after_test_gate_with_nothing_mature_is_not_applicable() -> None:
    with recording_bus() as sink:
        (gate,) = _gates(
            [{"metric": "roc_auc", "threshold": 0.6, "source": "out_of_time", "period": "window"}],
            None,
        )
    assert gate.passed and not gate.applicable and gate.period == "window"
    assert any("gate not applicable" in m for m in sink.messages())


def test_after_test_gates_can_be_selected_or_skipped() -> None:
    report = _report(_cell("month", "2026-06", 0.9))
    gates = [
        {"metric": "roc_auc", "threshold": 0.6},
        {"metric": "roc_auc", "threshold": 0.6, "source": "out_of_time", "min_rows": 1},
    ]
    assert [g.metric for g in _gates(gates, report, out_of_time="only")] == ["oot_roc_auc"]
    assert [g.metric for g in _gates(gates, report, out_of_time="skip")] == ["roc_auc"]


def _stability_cell(period: str, key: str, value: float, n_rows: int = 200) -> StabilityCell:
    stat = ShiftStat(method="psi", value=value, n_current=n_rows, n_baseline=100)
    return StabilityCell(
        period=period,  # type: ignore[arg-type]
        key=key,
        slot="all",
        n_rows=n_rows,
        reference_rows=100,
        gate=MonitorStats(prediction_shift=stat, feature_shift={"x": stat}),
    )


def test_stability_judges_each_cell_with_the_monitor_rules() -> None:
    spec = StabilitySpec.model_validate(
        {"prediction_shift": {"threshold": 0.25}, "feature_shift": {"threshold": 0.25}}
    )
    report = _report(
        stability=[
            _stability_cell("month", "2026-05", 0.05),
            _stability_cell("month", "2026-06", 0.4),
            _stability_cell("month", "2026-07", 0.9, n_rows=5),  # too small to judge
            _stability_cell("window", "window", 0.9),  # another grain
        ]
    )
    with recording_bus() as sink:
        outcome = evaluate_stability(spec, report, resource="model.demo.m")
    assert outcome.declared and outcome.judged
    results = outcome.results
    subjects = {(r.subject, r.passed) for r in results}
    assert ("2026-05", True) in subjects and ("2026-06: x", False) in subjects
    assert not any("2026-07" in (r.subject or "") for r in results)
    failing = next(r for r in results if not r.passed and r.monitor == "prediction_shift")
    assert failing.message and failing.message.startswith("[2026-06] ")
    assert any(m == "stability 2026-06: 200 rows against 100 test rows" for m in sink.messages())


def test_stability_with_nothing_to_judge_warns() -> None:
    """Three states, not two (C-1).

    "not declared" and "declared but nothing mature" both used to return ``[]``,
    and the caller read ``not stability`` as if it meant one - while that
    distinction is exactly what separates a ``not_gated`` verdict from a
    ``true`` one.
    """
    spec = StabilitySpec.model_validate({"prediction_shift": {"threshold": 0.25}})
    undeclared = evaluate_stability(None, None, resource="m")
    assert (undeclared.declared, undeclared.judged, undeclared.results) == (False, False, [])
    with recording_bus() as sink:
        nothing_mature = evaluate_stability(spec, None, resource="m")
    assert (nothing_mature.declared, nothing_mature.judged) == (True, False)
    assert nothing_mature.results == []
    assert any("stability not judged" in m for m in sink.messages())
    # and the ambiguity cannot be reintroduced by a truthiness test
    with pytest.raises(TypeError, match="three states"):
        bool(nothing_mature)


def _spec(**evaluation: Any) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": "binary_classification",
            "adapter": "fake",
            "owner": "o@example.com",
            "dataset": "ref('d')",
            "target": "y",
            "seed": 1,
            "evaluation": {"protocol": {"split": "temporal"}, "metrics": ["roc_auc"], **evaluation},
        }
    )


def test_the_after_test_verdict() -> None:
    oot = {"metric": "roc_auc", "threshold": 0.5, "source": "out_of_time"}
    # source is THE discriminator now, not period (C-1)
    judged = GateResult(
        metric="oot_roc_auc", kind="threshold", passed=True, period="month", source="out_of_time"
    )
    unjudged = judged.model_copy(update={"applicable": False})
    failed = judged.model_copy(update={"passed": False})
    breach = MonitorResult(
        monitor="prediction_shift", measure="psi", threshold=0.2, passed=False, value=0.3
    )
    none_declared = StabilityOutcome(declared=False)
    assert after_test_verdict(_spec(), [], none_declared) is None
    assert after_test_verdict(_spec(gates=[oot]), [unjudged], none_declared) == "not_gated"
    assert after_test_verdict(_spec(gates=[oot]), [judged], none_declared) == "true"
    assert after_test_verdict(_spec(gates=[oot]), [failed], none_declared) == "false"
    stability = {"prediction_shift": {"threshold": 0.2}}
    breached = StabilityOutcome(declared=True, results=[breach])
    assert after_test_verdict(_spec(stability=stability), [], breached) == "false"
    # a gate whose period is set but whose source is NOT out_of_time is not an
    # after-test gate: the old `period is not None` test could not say so
    other_kind = GateResult(
        metric="roc_auc", kind="threshold", passed=False, period="month", source="test"
    )
    assert after_test_verdict(_spec(gates=[oot]), [judged, other_kind], none_declared) == "true"
    assert after_test_tags("true", "a", "build") == {
        "mbt.oot_check.passed": "true",
        "mbt.oot_check.anchor": "a",
        "mbt.oot_check.source": "build",
    }


def test_gate_failure_summary_names_the_cell_and_stability() -> None:
    gate = GateResult(
        metric="oot_roc_auc",
        kind="threshold",
        passed=False,
        expected=0.7,
        actual=0.61,
        period="month",
        cell="2026-06",
    )
    breach = MonitorResult(
        monitor="prediction_shift",
        subject="2026-06",
        measure="psi",
        threshold=0.2,
        passed=False,
        message="[2026-06] prediction psi=0.3 exceeds 0.2",
    )
    quiet = breach.model_copy(update={"message": None})
    summary = gate_failure_summary([gate], [breach, quiet])
    assert summary == (
        "gate breach: oot_roc_auc [2026-06]=0.6100 failed threshold 0.7; "
        "stability [2026-06] prediction psi=0.3 exceeds 0.2; stability 2026-06"
    )


# -- node logs ------------------------------------------------------------------------------


def test_node_logs_file_each_event_under_its_node(tmp_path: Path) -> None:
    sink = NodeLogSink(tmp_path / "logs")
    sink.write(LogMessage(message="compiled"))
    sink.write(LogMessage(unique_id="dataset.d", message="built"))
    with node_scope("model.m"):
        sink.write(LogMessage(level="debug", message="a framework line"))
    sink.write(LogMessage(unique_id="model.m", message="password=hunter2"))
    assert (tmp_path / "logs" / INVOCATION_LOG).read_text().endswith("compiled\n")
    model_log = sink.read("model.m")
    assert "DEBUG a framework line" in model_log
    assert sink.read("model.nothing") == ""
    text = combined_log(sink, ["dataset.d", "dataset.empty", "model.m"], header=["title"])
    assert text.startswith("# title\n\n## dataset.d\n")
    assert "## dataset.empty" not in text
    sink.close()


# -- flat parameters -----------------------------------------------------------------------


def test_flatten_encodes_what_a_key_cannot_hold() -> None:
    params = flatten(
        "model",
        {
            "features": {"transforms": {"tenure days": {"cap": 1}}, "include": ["*"]},
            "tuning": None,
            "hyperparameters": {},
            "odd": {"has space?": 1},
            "flag": True,
        },
    )
    assert params == {
        "model.features.include": '["*"]',
        "model.features.transforms": '{"tenure days":{"cap":1}}',
        "model.flag": "true",
        "model.hyperparameters": "{}",
        "model.odd": '{"has space?":1}',
        "model.tuning": "null",
    }
    assert model_params({"name": "m"}, ["a", "b"])["model.resolved.n_features"] == "2"
    dataset = dataset_params(
        {"name": "d"},
        windows={"test": ["s", "e"]},
        anchor="a",
        sample_fraction=None,
        row_counts={"test": 3},
    )
    assert dataset == {
        "dataset.name": "d",
        "dataset.windows.test.start": "s",
        "dataset.windows.test.end": "e",
        "dataset.anchor": "a",
        "dataset.rows.test": "3",
    }


# -- the split router ------------------------------------------------------------------------


def test_the_split_router_serves_and_stages_each_split(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from mbt_adapter_base.materialization import (
        MaterializedDatasetHandle,
        write_materialization_metadata,
    )

    table = pa.table({"x": [1, 2]})
    pq.write_table(table, tmp_path / "test.parquet")
    write_materialization_metadata(
        tmp_path,
        snapshot_id="sha256:snap",
        dataset="d",
        label_column="y",
        time_column=None,
        windows={},
        sample_fraction=1.0,
        row_counts={"test": 2},
    )
    staged = MaterializedDatasetHandle(tmp_path)
    memory = InMemoryDatasetHandle({"out_of_time": pa.table({"x": [3]})})
    router = SplitRouter({"test": staged, "out_of_time": memory})
    assert router.splits() == {"test", "out_of_time"}
    assert router.snapshot_id == "sha256:snap"
    assert router.read("out_of_time").num_rows == 1
    assert router.split_path("test") == tmp_path / "test.parquet"
    assert pq.read_table(router.split_path("out_of_time")).num_rows == 1
    assert router.profile().n_rows == {"test": 2}
    assert router.locator().snapshot_id == "sha256:snap"
    assert staged.metadata["row_counts"] == {"test": 2}


# -- the model card --------------------------------------------------------------------------


def test_the_model_card_shows_after_test_gates_and_stability() -> None:
    from mbt.docsgen.generator import _gate_table, _stability_table

    result = NodeResult(
        unique_id="model.demo.m",
        status="success",
        gates=[
            GateResult(
                metric="oot_roc_auc",
                kind="threshold",
                passed=True,
                expected=0.6,
                actual=0.7,
                period="month",
                cell="2026-06",
            ),
            GateResult(
                metric="oot_pr_auc",
                kind="threshold",
                passed=True,
                expected=0.3,
                period="window",
                applicable=False,
            ),
        ],
        monitors=[
            MonitorResult(
                monitor="prediction_shift",
                subject="2026-06",
                measure="psi",
                threshold=0.25,
                passed=False,
                value=0.3,
            ),
            MonitorResult(monitor="feature_shift", measure="psi", threshold=0.25, passed=True),
        ],
    )
    gates = _gate_table(result)
    assert "threshold 0.6 (worst month at 2026-06)" in gates
    assert "threshold 0.3 (worst window)" in gates
    assert "not applicable" in gates and "N/A" in gates
    stability = _stability_table(result)
    assert "0.3000" in stability and "FAIL" in stability and "<td>-</td>" in stability
    assert _stability_table(None) == ""


@pytest.mark.parametrize("missing", ["directories", None])
def test_the_run_log_upload_falls_back_and_survives_errors(
    missing: str | None, tmp_path: Path
) -> None:
    from mbt.contracts import ManifestNode
    from mbt.execute.runners import ModelRunner

    sink = NodeLogSink(tmp_path)
    sink.write(LogMessage(unique_id="model.demo.m", message="trained"))
    documents: list[str] = []

    class _Tracker:
        def resume(self, run_id: str) -> Any:
            if missing is None:
                raise RuntimeError("tracker down")
            return SimpleNamespace(run_id=run_id)

        def log_document(self, run: Any, path: Path) -> None:
            documents.append(path.read_text())

    ctx = SimpleNamespace(
        node_logs=sink,
        command="build",
        run_id="r1",
        manifest=SimpleNamespace(metadata=SimpleNamespace(target="dev", anchor="a")),
        tracking=_Tracker,
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.ctx = ctx  # type: ignore[assignment]
    node = ManifestNode(
        unique_id="model.demo.m", resource_type="model", name="m", path="p", config={}
    )
    with recording_bus() as bus:
        runner._upload_log(node, "run-1")
        runner._upload_log(node, None)  # nothing to attach to
    if missing is None:
        assert any("could not upload the run log" in m for m in bus.messages())
    else:
        assert documents and "trained" in documents[0]


# -- the composition, directly (C-1) ----------------------------------------


def test_judge_composes_gates_stability_verdict_and_summary() -> None:
    """The five-call sequence, tested at the seam rather than through run_command.

    Each leaf had a tight unit test; the composition was reachable only through
    a full ``run_command``, and the composition is where the coupling was.
    """
    from mbt.quality.judgement import judge

    oot = {"metric": "roc_auc", "threshold": 0.5, "source": "out_of_time"}
    passing = GateResult(
        metric="oot_roc_auc", kind="threshold", passed=True, period="month", source="out_of_time"
    )
    outcome = judge(_spec(gates=[oot]), [passing], None, resource="m")
    assert outcome.passed is True
    assert outcome.verdict == "true"
    assert outcome.failure_summary is None
    assert outcome.stability.declared is False


def test_judge_reports_the_failure_and_the_verdict_together() -> None:
    from mbt.quality.judgement import judge

    oot = {"metric": "roc_auc", "threshold": 0.5, "source": "out_of_time"}
    failing = GateResult(
        metric="oot_roc_auc",
        kind="threshold",
        passed=False,
        expected=0.5,
        actual=0.4,
        period="month",
        source="out_of_time",
    )
    outcome = judge(_spec(gates=[oot]), [failing], None, resource="m")
    assert outcome.passed is False
    assert outcome.verdict == "false"
    assert outcome.failure_summary and "oot_roc_auc" in outcome.failure_summary


def test_judge_says_not_gated_when_nothing_was_mature_enough() -> None:
    """A gate that judged nothing proved nothing, and a promotion requiring the
    check must be able to tell that apart from a pass."""
    from mbt.quality.judgement import judge

    oot = {"metric": "roc_auc", "threshold": 0.5, "source": "out_of_time"}
    unjudged = GateResult(
        metric="oot_roc_auc",
        kind="threshold",
        passed=True,
        applicable=False,
        period="month",
        source="out_of_time",
    )
    outcome = judge(_spec(gates=[oot]), [unjudged], None, resource="m")
    assert outcome.passed is True  # an inapplicable gate does not fail the run
    assert outcome.verdict == "not_gated"  # but it did not prove anything either
