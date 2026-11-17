"""The training-job entrypoint: ``python -m mbt.execute.job <job.json>`` (TSD §10.3).

Everything framework- or data-heavy happens here, inside a subprocess (or a
remote job in v1): hooks, AUTO resolution, tuning, training, metric
computation for challenger *and* champion, artifact export, tracking logging.
No registry access, no gate decisions - core compares, jobs compute.

The result is written to ``<job.json>.result.json``; stdout carries the
JSON event stream the coordinator forwards.

This module is the entrypoint and the dispatch; the work is one module per
step (FEEDBACK v7, "Smaller notes": it was 1,650 lines). Their leading
underscores mark them private to the job subsystem, not to a file:

- ``job_setup``: re-render adapter config from the job's own environment, build the runtime
- ``job_carve``: validation and calibration slices carved from the train split
- ``job_metrics``: metric computation shared by tuning, training and evaluation
- ``job_tuning``: hyperparameter search, with the robust objective
- ``job_backtest``: rolling and walk-forward backtests
- ``job_train``: tune, fit, calibrate, evaluate, export, log
- ``job_score``: scoring, evaluation and the after-test check
"""

import contextlib
import sys
import traceback
from pathlib import Path
from typing import Any

from mbt.events import EventBus, JsonLinesSink, get_bus, set_bus
from mbt.exceptions import MbtError
from mbt.execute.job_score import _run_evaluate, _run_oot_check, _run_score
from mbt.execute.job_setup import _prepare, _tracking_adapter
from mbt.execute.job_train import _run_train
from mbt_adapter_base import (
    JobResult,
    TrainingJob,
)


def run_job(job: TrainingJob) -> JobResult:
    """Execute one training/evaluation/scoring job; never raises (returns errors)."""
    tracking = None
    run_handle = None
    try:
        if job.mode == "score":
            return _run_score(job)
        runtime = _prepare(job)
        if job.tracking is not None and job.mode == "train":
            tracking = _tracking_adapter(job)
            run_handle = tracking.start_run(job.node, dict(job.tracking_meta))

        if job.mode == "evaluate":
            return _run_evaluate(runtime)
        if job.mode == "oot_check":
            return _run_oot_check(runtime)

        result = _run_train(runtime, tracking, run_handle)
        if tracking is not None and run_handle is not None:
            tracking.end_run(run_handle, "FINISHED")
        return result
    except MbtError as exc:
        _best_effort_fail(tracking, run_handle)
        return JobResult(status="error", error=str(exc), tracking_run_id=_run_id(run_handle))
    except Exception as exc:
        _best_effort_fail(tracking, run_handle)
        tail = traceback.format_exc(limit=8)
        return JobResult(
            status="error", error=f"{exc!r}\n{tail}", tracking_run_id=_run_id(run_handle)
        )


def _run_id(run_handle: Any) -> str | None:
    """The failed run's id, so the coordinator can still attach its log."""
    return str(run_handle.run_id) if run_handle is not None else None


# -- metric computation ---------------------------------------------------------


# -- tuning (TSD §13.5, ADR-8) ----------------------------------------------------


# -- main paths -------------------------------------------------------------------


def _best_effort_fail(tracking: Any, run_handle: Any) -> None:
    if tracking is not None and run_handle is not None:
        with contextlib.suppress(Exception):
            tracking.end_run(run_handle, "FAILED")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stderr.write("usage: python -m mbt.execute.job <job.json>\n")
        return 2
    job_path = Path(args[0])
    set_bus(EventBus(sinks=[JsonLinesSink(sys.stdout)]))
    job = TrainingJob.model_validate_json(job_path.read_text())
    get_bus().run_id = job.run_id
    result = run_job(job)
    from mbt.adapters.local.compute import result_path_for
    from mbt.secrets import redact

    # The job taints its rendered adapter configs, so an exception message
    # embedding a credential (connection strings in tracebacks) is masked
    # before it ever reaches disk - same guarantee as the event sinks.
    result_path_for(job_path).write_text(redact(result.model_dump_json()))
    return 0 if result.status == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
