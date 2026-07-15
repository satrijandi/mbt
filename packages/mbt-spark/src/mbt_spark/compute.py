"""Spark ComputeAdapter: run mbt training jobs under spark-submit (ADR-3).

The serialized ``TrainingJob`` is exactly the seam remote compute reuses:
``submit`` spark-submits a tiny wrapper that runs ``mbt.execute.job`` on
the driver; the result comes back through the standard result file, and
the job's JSON event stream on stdout is forwarded to the mbt event bus.

The cluster's Python environment must have mbt-core plus the model's
training adapter installed (the same image contract K8s adapters use,
TSD §22). ``master: local[*]`` makes this adapter fully testable without
a cluster - and is itself useful for memory-isolated local runs.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from mbt_adapter_base import JobResult, TrainingJob


@dataclass
class SparkJobHandle:
    job_id: str
    process: "subprocess.Popen[str]"
    job_path: Path
    #: Why terminate() was called (timeout, --fail-fast); None if never.
    terminated_reason: str | None = None


class SparkComputeAdapter:
    """ComputeAdapter delegating jobs to spark-submit."""

    name = "spark"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Same key and validation as the local adapter (the public seam).
        from mbt.adapters.local.compute import parse_job_timeout

        config = config or {}
        self.master: str = str(config.get("master", "local[*]"))
        self.deploy_mode: str | None = config.get("deploy_mode")
        self.conf: dict[str, Any] = dict(config.get("conf", {}))
        self.spark_submit: str = str(config.get("spark_submit", "spark-submit"))
        self.job_timeout_seconds = parse_job_timeout(config)

    def _command(self, job_path: Path) -> list[str]:
        wrapper = files("mbt_spark") / "job_wrapper.py"
        command = [self.spark_submit, "--master", self.master]
        if self.deploy_mode:
            command += ["--deploy-mode", self.deploy_mode]
        for key, value in self.conf.items():
            command += ["--conf", f"{key}={value}"]
        command += [str(wrapper), str(job_path)]
        return command

    def submit(self, job: TrainingJob) -> SparkJobHandle:
        job_dir = Path(tempfile.mkdtemp(prefix="mbt-spark-job-"))
        job_path = job_dir / "job.json"
        job_path.write_text(job.model_dump_json())
        env = dict(os.environ)
        # the driver must run the same interpreter/venv as the coordinator
        env.setdefault("PYSPARK_PYTHON", sys.executable)
        env.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        process = subprocess.Popen(
            self._command(job_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=job.project_dir,
            env=env,
        )
        return SparkJobHandle(
            job_id=f"spark-{uuid.uuid4().hex[:12]}", process=process, job_path=job_path
        )

    def terminate(self, handle: SparkJobHandle, reason: str = "terminated") -> None:
        """SIGTERM spark-submit (its shutdown hook tears the driver down),
        then SIGKILL after a grace period; mirrors the local adapter."""
        handle.terminated_reason = reason
        process = handle.process
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def wait(self, handle: SparkJobHandle) -> JobResult:
        # Forward the job's event stream through the mbt bus, exactly like
        # the local compute adapter (spark-submit noise stays at debug level).
        from mbt.adapters.local.compute import parse_job_line, result_path_for
        from mbt.events import get_bus

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
                if line:
                    bus.emit(parse_job_line(line))
            returncode = handle.process.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()

        result_path = result_path_for(handle.job_path)
        if result_path.is_file():
            result = JobResult.model_validate_json(result_path.read_text())
            if result.status != "error":
                shutil.rmtree(handle.job_path.parent, ignore_errors=True)
            return result
        if handle.terminated_reason is not None:
            return JobResult(
                status="error",
                error=(
                    f"spark-submit job {handle.terminated_reason} "
                    f"(job payload kept at {handle.job_path})"
                ),
            )
        return JobResult(
            status="error",
            error=(
                f"spark-submit exited with code {returncode} without writing a "
                f"result (job payload kept at {handle.job_path}); check that the "
                "driver environment has mbt-core and the training adapter installed"
            ),
        )
