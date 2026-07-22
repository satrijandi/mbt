"""`mbt predictions ls/show` read surface over the prediction store (R2-12)."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
from cli_unit_helpers import debug
from cli_unit_helpers import invoke as cli_invoke
from core_helpers import write
from exec_unit_helpers import SCORING_UID, make_options
from test_execution import invoke
from test_monitor_ground_truth import GT_SCORING_YML, _write_outcomes, monitor
from test_scoring_execution import SCORING_YML, _build_and_promote, _prediction_runs, _write_batch

from mbt.adapters.registry import AdapterRegistry
from mbt.execute.predictions_view import list_prediction_runs, show_prediction_run

GT_SOURCES_YML = """
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
      - name: scoring_batch
        path: data/scoring_batch/*.parquet
      - name: churn_outcomes
        path: data/churn_outcomes/*.parquet
"""

SCORING_SOURCES_YML = """
sources:
  - name: lakehouse
    tables:
      - name: subscribers
        path: data/subscribers/*.parquet
      - name: scoring_batch
        path: data/scoring_batch/*.parquet
"""


@pytest.fixture()
def gt_project(demo_project: Path) -> Path:
    write(demo_project / "sources.yml", GT_SOURCES_YML)
    write(demo_project / "scoring/churn_scoring.yml", GT_SCORING_YML)
    _write_batch(demo_project)
    _write_outcomes(demo_project)
    return demo_project


@pytest.fixture()
def scoring_project(demo_project: Path) -> Path:
    write(demo_project / "sources.yml", SCORING_SOURCES_YML)
    write(demo_project / "scoring/churn_scoring.yml", SCORING_YML)
    _write_batch(demo_project)
    return demo_project


def _opts(project_dir: Path, **kwargs: object):
    return make_options(project_dir, command="predictions", **kwargs)


def test_lists_a_scored_run_as_not_yet_matured(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0

    runs = list_prediction_runs(_opts(gt_project), registry=fake_registry)
    assert len(runs) == 1
    run = runs[0]
    assert run.scoring == SCORING_UID
    assert run.model_name == "churn_model" and run.model_version == "1"
    assert run.row_count > 0
    assert run.matured is False  # maturity 14d, anchored at the scoring time
    assert not run.evaluated and run.realized == {} and run.coverage is None

    # show finds it; a bogus key does not
    assert show_prediction_run(_opts(gt_project), run.run_key, registry=fake_registry) is not None
    assert show_prediction_run(_opts(gt_project), "nope", registry=fake_registry) is None


def test_evaluated_run_reports_realized_metrics(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    monitor(gt_project, fake_registry)  # matures + evaluates at +20d

    from core_helpers import TEST_ANCHOR

    matured_opts = _opts(gt_project, anchor=TEST_ANCHOR + timedelta(days=20))
    (run,) = list_prediction_runs(matured_opts, registry=fake_registry)
    assert run.matured is True and run.evaluated
    assert run.realized and run.coverage is not None


def test_unparseable_scored_at_reports_matured_unknown(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    sidecar = _prediction_runs(gt_project)[0] / "predictions.json"
    info = json.loads(sidecar.read_text())
    info["scored_at"] = "not-a-timestamp"
    sidecar.write_text(json.dumps(info))

    (run,) = list_prediction_runs(_opts(gt_project), registry=fake_registry)
    assert run.matured is None  # a ground_truth node, but the timestamp is unreadable


def test_scoring_node_without_ground_truth_has_no_maturity(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    (run,) = list_prediction_runs(_opts(scoring_project), registry=fake_registry)
    assert run.matured is None  # no ground_truth block -> maturity is undefined


def test_zero_row_run_is_not_reported_as_matured(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    # `mbt monitor` skips a 0-row run (it can never be evaluated), so `ls` must
    # not report it as matured=yes - that would read as matured / evaluated=no
    # forever. It reports maturity unknown (None), matching monitor's skip.
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0

    # rewrite the sidecar to a 0-row run whose scored_at is long past the 14d
    # maturity window (the OLD code would then report matured=True).
    (run_dir,) = _prediction_runs(gt_project)
    info = json.loads((run_dir / "predictions.json").read_text())
    info["row_count"] = 0
    info["scored_at"] = "2020-01-01T00:00:00Z"
    (run_dir / "predictions.json").write_text(json.dumps(info))

    (run,) = list_prediction_runs(_opts(gt_project), registry=fake_registry)
    assert run.row_count == 0
    assert run.matured is None  # not True: monitor will never evaluate it


def test_predictions_ls_reuses_the_built_manifest(
    scoring_project: Path, fake_registry: AdapterRegistry
) -> None:
    # ls is read-only: it reuses target/manifest.json rather than recompiling, so
    # it does not overwrite that build artifact and still works when a data
    # source is gone (a transient outage during an incident must not fail an
    # inspection command).
    import shutil

    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    manifest_file = scoring_project / "target" / "manifest.json"
    before = manifest_file.read_text()

    shutil.rmtree(scoring_project / "data" / "scoring_batch")  # simulate a source outage

    runs = list_prediction_runs(_opts(scoring_project), registry=fake_registry)
    assert len(runs) == 1  # ls still works without the source
    assert manifest_file.read_text() == before  # the build artifact is untouched

    # an explicit --manifest also reads (never recompiles), and still needs no
    # source data (it verifies the environment, not the sources).
    (via_manifest,) = list_prediction_runs(
        _opts(scoring_project, manifest_path=str(manifest_file)), registry=fake_registry
    )
    assert via_manifest.row_count > 0


def test_no_runs_before_scoring(scoring_project: Path, fake_registry: AdapterRegistry) -> None:
    _build_and_promote(scoring_project, fake_registry)  # built + promoted, never scored
    assert list_prediction_runs(_opts(scoring_project), registry=fake_registry) == []


# -- CLI rendering ---------------------------------------------------------------------


def test_cli_predictions_ls_and_show(scoring_project: Path, fake_registry: AdapterRegistry) -> None:
    _build_and_promote(scoring_project, fake_registry)
    assert invoke(scoring_project, fake_registry, "score").exit_code() == 0
    base = ["--project-dir", str(scoring_project)]

    table = cli_invoke(["predictions", "ls", *base])
    assert table.exit_code == 0, debug(table)
    assert "scoring" in table.output and "run_key" in table.output

    # -q suppresses the compile event lines so stdout is clean JSON
    payload = cli_invoke(["predictions", "ls", *base, "--output", "json", "-q"])
    assert payload.exit_code == 0, debug(payload)
    runs = json.loads(payload.output)
    assert len(runs) == 1
    run_key = runs[0]["run_key"]

    shown = cli_invoke(["predictions", "show", run_key, *base])
    assert shown.exit_code == 0, debug(shown)
    assert run_key in shown.output and "champion" in shown.output

    shown_json = cli_invoke(["predictions", "show", run_key, *base, "--output", "json", "-q"])
    assert shown_json.exit_code == 0 and json.loads(shown_json.output)["run_key"] == run_key

    missing = cli_invoke(["predictions", "show", "nope", *base])
    assert missing.exit_code == 1
    assert "no prediction run 'nope' found" in missing.stderr


def test_cli_predictions_ls_empty(scoring_project: Path, fake_registry: AdapterRegistry) -> None:
    _build_and_promote(scoring_project, fake_registry)  # never scored
    result = cli_invoke(["predictions", "ls", "--project-dir", str(scoring_project)])
    assert result.exit_code == 0, debug(result)
    assert "no prediction runs found" in result.output


def test_cli_predictions_show_evaluated_lists_realized(
    gt_project: Path, fake_registry: AdapterRegistry
) -> None:
    _build_and_promote(gt_project, fake_registry)
    assert invoke(gt_project, fake_registry, "score").exit_code() == 0
    monitor(gt_project, fake_registry)  # evaluate at +20d
    run_key = _prediction_runs(gt_project)[0].name
    # a fresh compile anchors at "now", well past the run's maturity; the
    # evaluated marker monitor wrote is anchor-independent, so realized shows
    result = cli_invoke(["predictions", "show", run_key, "--project-dir", str(gt_project)])
    assert result.exit_code == 0, debug(result)
    assert "realized" in result.output and "coverage" in result.output
