"""Local ComputeAdapter: training jobs as subprocesses (TSD §13.4, ADR-3).

``submit`` spawns ``python -m mbt.execute.job <job.json>``: crash/memory
isolation, real ``--threads`` parallelism, and the serialization seam K8s/Ray
reuse in v1. The payload carries env-var *names* only; the subprocess
inherits the environment and re-resolves values itself (TSD §18).

The job writes its ``JobResult`` to ``<job.json>.result.json`` - stdout
belongs to the event stream, which the coordinator forwards line by line.
"""

import json
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mbt.contracts import JobResult, TrainingJob
from mbt.events import get_bus
from mbt.events.models import Event, JobLine


def result_path_for(job_path: Path) -> Path:
    return job_path.with_suffix(job_path.suffix + ".result.json")


@dataclass
class LocalJobHandle:
    job_id: str
    process: subprocess.Popen[str]
    job_path: Path
    forwarded: list[str] = field(default_factory=list)


class LocalComputeAdapter:
    """Runs each training job in a fresh Python subprocess."""

    name = "local"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def submit(self, job: TrainingJob) -> LocalJobHandle:
        job_dir = Path(tempfile.mkdtemp(prefix="mbt-job-"))
        job_path = job_dir / "job.json"
        job_path.write_text(job.model_dump_json())
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "mbt.execute.job", str(job_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=job.project_dir,
        )
        return LocalJobHandle(
            job_id=f"local-{uuid.uuid4().hex[:12]}", process=process, job_path=job_path
        )

    def wait(self, handle: LocalJobHandle) -> JobResult:
        bus = get_bus()
        assert handle.process.stdout is not None
        for line in handle.process.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            bus.emit(_parse_job_line(line))
        returncode = handle.process.wait()

        result_path = result_path_for(handle.job_path)
        if result_path.is_file():
            try:
                return JobResult.model_validate_json(result_path.read_text())
            except ValidationError as exc:
                return JobResult(status="error", error=f"unreadable job result: {exc}")
        return JobResult(
            status="error",
            error=(
                f"training job exited with code {returncode} without writing a result "
                f"(job payload kept at {handle.job_path})"
            ),
        )


def _parse_job_line(line: str) -> Event:
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
