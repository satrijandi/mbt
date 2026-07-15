"""Local ComputeAdapter: training jobs as subprocesses (TSD §13.4, ADR-3).

``submit`` spawns ``python -m mbt.execute.job <job.json>``: crash/memory
isolation, real ``--threads`` parallelism, and the serialization seam K8s/Ray
reuse in v1. The payload carries env-var *names* only; the subprocess
inherits the environment and re-resolves values itself (TSD §18).

The job writes its ``JobResult`` to ``<job.json>.result.json`` - stdout
belongs to the event stream, which the coordinator forwards line by line.
Successful job payload dirs are removed after the result is read; error
payloads stay behind for debugging (the error message points at them).

Operational guard rails: ``job_timeout_seconds`` in the compute profile
config kills any job that outlives it, and ``terminate`` lets the scheduler
reclaim in-flight jobs on ``--fail-fast``.

``parse_job_line``, ``result_path_for``, and ``parse_job_timeout`` are
public API: remote compute adapters (mbt-spark today, K8s/Ray in v1) reuse
them to forward the same event stream, read the same result file, and
honor the same timeout config key.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mbt.contracts import JobResult, TrainingJob
from mbt.events import get_bus
from mbt.events.models import Event, JobLine
from mbt.exceptions import ConfigError


def result_path_for(job_path: Path) -> Path:
    return job_path.with_suffix(job_path.suffix + ".result.json")


def parse_job_timeout(config: dict[str, Any]) -> float | None:
    """Validate the ``job_timeout_seconds`` compute-config key (None = no limit)."""
    raw = config.get("job_timeout_seconds")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        raise ConfigError(
            f"compute config job_timeout_seconds must be a positive number, got {raw!r}",
            hint="seconds a single training job may run before it is killed; omit for no limit",
        )
    return value


@dataclass
class LocalJobHandle:
    job_id: str
    process: subprocess.Popen[str]
    job_path: Path
    #: Why terminate() was called (timeout, --fail-fast); None if never.
    terminated_reason: str | None = None


class LocalComputeAdapter:
    """Runs each training job in a fresh Python subprocess."""

    name = "local"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.job_timeout_seconds = parse_job_timeout(self.config)

    def submit(self, job: TrainingJob) -> LocalJobHandle:
        job_dir = Path(tempfile.mkdtemp(prefix="mbt-job-"))
        job_path = job_dir / "job.json"
        job_path.write_text(job.model_dump_json())
        process = subprocess.Popen(
            [sys.executable, "-m", "mbt.execute.job", str(job_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=job.project_dir,
        )
        return LocalJobHandle(
            job_id=f"local-{uuid.uuid4().hex[:12]}", process=process, job_path=job_path
        )

    def terminate(self, handle: LocalJobHandle, reason: str = "terminated") -> None:
        """SIGTERM, then SIGKILL after a short grace period.

        Called by the job-timeout watchdog and by the scheduler on
        ``--fail-fast``; a concurrent ``wait`` unblocks (stdout closes) and
        reports ``reason`` in its error result.
        """
        handle.terminated_reason = reason
        process = handle.process
        if process.poll() is not None:
            return  # already exited: the result file decides the outcome
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def wait(self, handle: LocalJobHandle) -> JobResult:
        bus = get_bus()
        timeout = self.job_timeout_seconds
        watchdog: threading.Timer | None = None
        if timeout is not None:
            watchdog = threading.Timer(
                timeout,
                self.terminate,
                args=(handle, f"timed out after {timeout:g}s and was killed"),
            )
            watchdog.daemon = True
            watchdog.start()
        assert handle.process.stdout is not None
        try:
            for raw_line in handle.process.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                bus.emit(parse_job_line(line))
            returncode = handle.process.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()

        result_path = result_path_for(handle.job_path)
        if result_path.is_file():
            try:
                result = JobResult.model_validate_json(result_path.read_text())
            except ValidationError as exc:
                return JobResult(status="error", error=f"unreadable job result: {exc}")
            if result.status != "error":
                # Served its purpose; error payloads stay for debugging.
                shutil.rmtree(handle.job_path.parent, ignore_errors=True)
            return result
        if handle.terminated_reason is not None:
            return JobResult(
                status="error",
                error=(
                    f"training job {handle.terminated_reason} "
                    f"(job payload kept at {handle.job_path})"
                ),
            )
        return JobResult(
            status="error",
            error=(
                f"training job exited with code {returncode} without writing a result "
                f"(job payload kept at {handle.job_path})"
            ),
        )


def parse_job_line(line: str) -> Event:
    """Rehydrate a forwarded JSON event line, or wrap raw output."""
    if line.startswith("{"):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return JobLine(line=line)
        if isinstance(payload, dict) and "event" in payload:
            from mbt.events import models as event_models

            event_cls = getattr(event_models, str(payload["event"]), None)
            if isinstance(event_cls, type) and issubclass(event_cls, Event):
                try:
                    return event_cls.model_validate(payload)
                except ValidationError:
                    return JobLine(line=line)
        return JobLine(line=line)
    return JobLine(line=line)
