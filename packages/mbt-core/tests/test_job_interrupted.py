"""A job that dies of the user's Ctrl-C is INTERRUPTED, not an ERROR (FEEDBACK v6 B-5)."""

from pathlib import Path
from typing import Any

import pytest
from exec_unit_helpers import MODEL_UID, make_training_job, recording_bus

from mbt.adapters.registry import AdapterRegistry
from mbt.events.models import NodeFinished
from mbt.exceptions import JobInterrupted
from mbt.execute.runners import ModelRunner
from mbt_adapter_base import JobResult, interrupted_job_result


class _InterruptedCompute:
    def submit(self, job: Any) -> str:
        return "handle"

    def wait(self, handle: str) -> JobResult:
        result = interrupted_job_result(-2)
        assert result is not None
        return result


def test_an_interrupted_job_reports_interrupted_and_re_raises(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, _ = make_training_job(demo_project, fake_registry)
    monkeypatch.setattr(ctx, "compute", _InterruptedCompute())
    with recording_bus() as sink, pytest.raises(JobInterrupted):
        ModelRunner(ctx).run(MODEL_UID)
    (finished,) = [e for e in sink.events if isinstance(e, NodeFinished)]
    assert finished.status == "interrupted" and finished.unique_id == MODEL_UID
    assert finished.human().split(" ")[1] == "INTERRUPTED"
    assert isinstance(JobInterrupted("x"), KeyboardInterrupt)  # exits 130 like Ctrl-C


def test_only_sigint_counts_as_an_interruption() -> None:
    assert interrupted_job_result(-2) is not None
    for code in (None, 0, 1, -9, -15):
        assert interrupted_job_result(code) is None
