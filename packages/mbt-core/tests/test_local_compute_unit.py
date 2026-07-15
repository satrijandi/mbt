"""Local compute adapter: subprocess payloads, event forwarding, results (ADR-3)."""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from misc_unit_helpers import RecordingSink, make_node

from mbt.adapters.local.compute import (
    LocalComputeAdapter,
    LocalJobHandle,
    parse_job_line,
    parse_job_timeout,
    result_path_for,
)
from mbt.contracts import AdapterRef, DatasetLocator, JobResult, TrainingJob
from mbt.events.bus import EventBus
from mbt.events.models import JobLine, LogMessage
from mbt.exceptions import ConfigError


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


def _job_dir(tmp_path: Path) -> Path:
    job_dir = tmp_path / "mbt-job-x"
    job_dir.mkdir()
    (job_dir / "job.json").write_text("{}")
    return job_dir


def test_wait_forwards_events_reads_the_result_and_cleans_up(
    tmp_path: Path, recording_bus: RecordingSink
) -> None:
    job_path = _job_dir(tmp_path) / "job.json"
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
    # A successful payload dir has served its purpose: no tmp accumulation.
    assert not job_path.parent.exists()


def test_wait_keeps_the_payload_dir_of_errored_jobs(
    tmp_path: Path, recording_bus: RecordingSink
) -> None:
    job_path = _job_dir(tmp_path) / "job.json"
    result_path_for(job_path).write_text(JobResult(status="error", error="boom").model_dump_json())
    handle = LocalJobHandle(job_id="local-1e", process=_FakeProcess([]), job_path=job_path)

    result = LocalComputeAdapter().wait(handle)
    assert result.status == "error"
    assert job_path.parent.is_dir()  # kept for debugging


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


def _sleeper_handle(tmp_path: Path) -> LocalJobHandle:
    """A real subprocess that would outlive any test unless killed."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return LocalJobHandle(
        job_id="local-sleeper", process=process, job_path=_job_dir(tmp_path) / "job.json"
    )


def test_wait_kills_a_job_that_outlives_the_timeout(
    tmp_path: Path, recording_bus: RecordingSink
) -> None:
    handle = _sleeper_handle(tmp_path)
    result = LocalComputeAdapter({"job_timeout_seconds": 0.5}).wait(handle)
    assert result.status == "error"
    assert result.error is not None
    assert "timed out after 0.5s and was killed" in result.error
    assert str(handle.job_path) in result.error  # payload kept for debugging
    assert handle.process.poll() is not None  # the subprocess is actually gone


def test_terminate_reports_the_reason_in_the_result(
    tmp_path: Path, recording_bus: RecordingSink
) -> None:
    adapter = LocalComputeAdapter()
    handle = _sleeper_handle(tmp_path)
    adapter.terminate(handle, "cancelled by --fail-fast")
    result = adapter.wait(handle)
    assert result.status == "error"
    assert result.error is not None and "cancelled by --fail-fast" in result.error
    assert handle.process.poll() is not None


@pytest.mark.parametrize("raw", [0, -3, "abc"])
def test_job_timeout_config_must_be_a_positive_number(raw: object) -> None:
    with pytest.raises(ConfigError, match="job_timeout_seconds"):
        LocalComputeAdapter({"job_timeout_seconds": raw})
    assert parse_job_timeout({}) is None
    assert parse_job_timeout({"job_timeout_seconds": "2.5"}) == 2.5


def test_parse_job_line_rehydrates_known_events() -> None:
    event = parse_job_line(json.dumps({"event": "LogMessage", "message": "hi"}))
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
    event = parse_job_line(line)
    assert isinstance(event, JobLine)
    assert event.line == line
