"""Local compute adapter: subprocess payloads, event forwarding, results (ADR-3)."""

import io
import json
import sys
from pathlib import Path

import pytest
from misc_unit_helpers import RecordingSink, make_node

from mbt.adapters.local.compute import (
    LocalComputeAdapter,
    LocalJobHandle,
    _parse_job_line,
    result_path_for,
)
from mbt.contracts import AdapterRef, DatasetLocator, JobResult, TrainingJob
from mbt.events.bus import EventBus
from mbt.events.models import JobLine, LogMessage


def _job(project_dir: Path) -> TrainingJob:
    return TrainingJob(
        run_id="run-1",
        project_dir=str(project_dir),
        target_name="dev",
        node=make_node("model.demo.churn"),
        dataset=DatasetLocator(adapter="local", uri="file:///data/ds", snapshot_id="sha256:abc"),
        data=AdapterRef(adapter="local", config={}),
    )


class _FakeProcess:
    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


@pytest.fixture()
def recording_bus(monkeypatch: pytest.MonkeyPatch) -> RecordingSink:
    sink = RecordingSink()
    monkeypatch.setattr("mbt.events.bus._bus", EventBus([sink]))
    return sink


def test_result_path_sits_next_to_the_job_payload() -> None:
    assert result_path_for(Path("/jobs/job.json")) == Path("/jobs/job.json.result.json")


def test_config_defaults_to_empty_dict() -> None:
    assert LocalComputeAdapter().config == {}
    assert LocalComputeAdapter({"threads": 2}).config == {"threads": 2}


def test_submit_serializes_the_job_and_spawns_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "jobdir"
    job_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    captured: dict = {}
    process = _FakeProcess([])

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr("mbt.adapters.local.compute.tempfile.mkdtemp", lambda prefix: str(job_dir))
    monkeypatch.setattr("mbt.adapters.local.compute.subprocess.Popen", fake_popen)

    job = _job(project_dir)
    handle = LocalComputeAdapter().submit(job)
    assert handle.job_id.startswith("local-")
    assert handle.process is process
    assert handle.job_path == job_dir / "job.json"
    assert TrainingJob.model_validate_json(handle.job_path.read_text()) == job
    assert captured["cmd"] == [sys.executable, "-m", "mbt.execute.job", str(handle.job_path)]
    assert captured["kwargs"]["cwd"] == str(project_dir)
    assert captured["kwargs"]["text"] is True


def test_wait_forwards_events_and_reads_the_result(
    tmp_path: Path, recording_bus: RecordingSink
) -> None:
    job_path = tmp_path / "job.json"
    result_path_for(job_path).write_text(JobResult(status="success").model_dump_json())
    lines = [
        json.dumps({"event": "LogMessage", "message": "from the job"}),
        "",  # blank lines are skipped
        "raw progress line",
    ]
    handle = LocalJobHandle(job_id="local-1", process=_FakeProcess(lines), job_path=job_path)

    result = LocalComputeAdapter().wait(handle)
    assert result.status == "success"
    assert [type(e).__name__ for e in recording_bus.events] == ["LogMessage", "JobLine"]
    assert recording_bus.events[0].message == "from the job"
    assert recording_bus.events[1].line == "raw progress line"


def test_wait_wraps_an_unreadable_result_file(tmp_path: Path, recording_bus: RecordingSink) -> None:
    job_path = tmp_path / "job.json"
    result_path_for(job_path).write_text('{"status": "success", "unknown_field": 1}')
    handle = LocalJobHandle(job_id="local-2", process=_FakeProcess([]), job_path=job_path)

    result = LocalComputeAdapter().wait(handle)
    assert result.status == "error"
    assert result.error is not None and "unreadable job result" in result.error


def test_wait_reports_a_missing_result_file_with_the_exit_code(
    tmp_path: Path, recording_bus: RecordingSink
) -> None:
    job_path = tmp_path / "job.json"
    handle = LocalJobHandle(
        job_id="local-3", process=_FakeProcess([], returncode=3), job_path=job_path
    )

    result = LocalComputeAdapter().wait(handle)
    assert result.status == "error"
    assert result.error is not None
    assert "exited with code 3" in result.error
    assert str(job_path) in result.error


def test_parse_job_line_rehydrates_known_events() -> None:
    event = _parse_job_line(json.dumps({"event": "LogMessage", "message": "hi"}))
    assert isinstance(event, LogMessage)
    assert event.message == "hi"


@pytest.mark.parametrize(
    "line",
    [
        "{not valid json",  # unparseable JSON stays raw
        '{"event": "LogMessage", "unknown_field": 1}',  # fails event validation
        '{"event": "NoSuchEvent"}',  # names no event class
        '{"event": "datetime"}',  # names a non-Event type in the module
        '{"message": "no event key"}',  # JSON without an event marker
        "plain text line",  # not JSON at all
    ],
)
def test_parse_job_line_wraps_everything_else_as_raw_output(line: str) -> None:
    event = _parse_job_line(line)
    assert isinstance(event, JobLine)
    assert event.line == line
