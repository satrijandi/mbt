"""Where an mbt process runs: on the worker, or in a container (ADR-34).

An operator builds the argument vector and decides what its exit code means; a
runner only starts that vector somewhere, hands back the exit code, the tail of
stderr, and the few files the operator asked for, and stops it when Airflow kills
the task. That is the whole seam, so a third place to run (a Kubernetes pod, say)
is one more class here and not a change to any operator.

Files come back through the runner because the worker does not necessarily see
them: a container's workspace is a path in the container, and a scheduler that
launches containers through the Docker socket usually has no mount of its own.
"""

import io
import os
import posixpath
import shlex
import shutil
import subprocess
import tarfile
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from airflow.sdk.exceptions import AirflowException, AirflowFailException

#: How many trailing stderr lines a hard-error message quotes.
TAIL_LINES = 4


@dataclass(frozen=True)
class DockerUnit:
    """The container one mbt process runs in."""

    #: The image, normally pinned by digest: it is the code that runs.
    image: str
    #: The docker network to join (the services the project talks to live on it).
    network: str | None = None
    #: Host path -> path in the container, mounted read-write. The run's workspace
    #: is one of these: every task of a run reads what the others wrote there.
    volumes: Mapping[str, str] = field(default_factory=dict)
    #: The container's working directory, where the project is.
    working_dir: str | None = None
    #: The container's environment. The worker's own is NOT inherited: a container
    #: starts clean, so everything it needs (credentials included) is named here.
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MbtDocker:
    """Run every mbt process in a container started through the Docker API.

    ``unit`` is a ``DockerUnit`` or a function that returns one. A function is
    called when the task runs, not when the DAG file is parsed, so a pin that
    changes between the two (a deploy repo that git-sync is still copying) is the
    one the task uses, and a missing pin fails the task with your own message.
    """

    unit: DockerUnit | Callable[[], DockerUnit]
    #: Seconds docker-py waits on one API call. Its default of 60 is shorter than a
    #: container create can take on a busy daemon, which failed tasks before mbt ran.
    api_timeout: int = 600

    def resolve(self) -> DockerUnit:
        return self.unit() if callable(self.unit) else self.unit


@dataclass(frozen=True)
class Completed:
    """What a finished process leaves behind."""

    code: int
    #: The last lines of stderr, which is where mbt's own words are.
    tail: tuple[str, ...]
    #: Standard output, only when it was asked for.
    stdout: str = ""
    #: The requested files that exist, by the path they were requested with.
    files: Mapping[str, bytes] = field(default_factory=dict)


class Runner(Protocol):
    """Start an argument vector somewhere, and report how it ended."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        log: Any,
        collect: Sequence[str] = (),
        capture_stdout: bool = False,
    ) -> Completed: ...

    def stop(self) -> None:
        """Ask the running process to stop (SIGTERM, which mbt treats as an interrupt)."""

    def remove_tree(self, path: str, *, log: Any) -> None:
        """Delete a directory tree as this runner sees it."""


class LocalRunner:
    """``mbt`` is an executable on the worker: a subprocess, streamed as it runs."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        log: Any,
        collect: Sequence[str] = (),
        capture_stdout: bool = False,
    ) -> Completed:
        """Events go to stderr and stdout is command data, which a task has no use for
        (its outcome is ``run_results.json``) unless the caller asks for it."""
        log.info("$ %s", shlex.join(argv))
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        try:
            proc = subprocess.Popen(
                list(argv),
                stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, **env},
            )
        except FileNotFoundError as exc:
            # Deterministic: retrying cannot make an executable appear.
            raise AirflowFailException(
                f"cannot run {argv[0]!r}: {exc.strerror}. Install mbt on the worker, "
                "or point MbtExecution.mbt_executable at it"
            ) from exc
        self._proc = proc
        assert proc.stderr is not None  # PIPE above
        stdout = ""
        if capture_stdout:
            # Both pipes are read at once, so a chatty stdout cannot block stderr.
            stdout, err = proc.communicate()
            lines: Iterable[str] = err.splitlines()
        else:
            lines = proc.stderr
        for raw in lines:
            line = raw.rstrip()
            if line:
                log.info("  %s", line)
                tail.append(line)
        code = proc.wait()
        files = {path: Path(path).read_bytes() for path in collect if Path(path).is_file()}
        return Completed(code, tuple(tail), stdout, files)

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def remove_tree(self, path: str, *, log: Any) -> None:
        shutil.rmtree(path, ignore_errors=True)
        log.info("removed %s", path)


def _docker_sdk() -> Any:
    """The Docker SDK, imported only when a container is actually started."""
    try:
        import docker
        import docker.errors
    except ImportError as exc:
        raise AirflowFailException(
            "MbtDocker needs the Docker SDK on the worker: pip install 'mbt-airflow[docker]'"
        ) from exc
    return docker


class DockerRunner:
    """Each process is one container: started, awaited, read back, and removed."""

    def __init__(self, config: MbtDocker) -> None:
        self.config = config
        self._container: Any = None

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        log: Any,
        collect: Sequence[str] = (),
        capture_stdout: bool = False,
    ) -> Completed:
        sdk = _docker_sdk()
        unit = self.config.resolve()
        log.info("$ %s   [in %s]", shlex.join(argv), unit.image)
        client = sdk.from_env(timeout=self.config.api_timeout)
        try:
            try:
                container = client.containers.run(
                    unit.image,
                    list(argv),
                    detach=True,
                    network=unit.network,
                    working_dir=unit.working_dir,
                    volumes={
                        host: {"bind": inside, "mode": "rw"}
                        for host, inside in unit.volumes.items()
                    },
                    environment={**unit.env, **env},
                )
            except sdk.errors.DockerException as exc:
                # Retryable: an image that is not there yet, or a daemon that is busy.
                raise AirflowException(f"cannot start {unit.image}: {exc}") from exc
            self._container = container
            try:
                code = container.wait().get("StatusCode", 1)
                # Output arrives when the container exits: a stream would hit the API
                # timeout on a quiet stretch of a long training.
                stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
                stdout = (
                    container.logs(stdout=True, stderr=False).decode(errors="replace")
                    if capture_stdout
                    else ""
                )
                files = {
                    path: data
                    for path in collect
                    if (data := self._read(container, path, sdk)) is not None
                }
            finally:
                self._container = None
                with suppress(sdk.errors.DockerException):
                    container.remove(force=True)
        finally:
            client.close()
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        for raw in stderr.replace("\r", "\n").split("\n"):
            line = raw.rstrip()
            if line:
                log.info("  %s", line)
                tail.append(line)
        return Completed(code, tuple(tail), stdout, files)

    @staticmethod
    def _read(container: Any, path: str, sdk: Any) -> bytes | None:
        """One file out of a stopped container, or None if it is not there."""
        try:
            chunks, _ = container.get_archive(path)
        except sdk.errors.NotFound:
            return None
        with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as archive:
            for member in archive:
                reader = archive.extractfile(member)
                if reader is not None:
                    return reader.read()
        return None

    def stop(self) -> None:
        container = self._container
        if container is not None:
            # SIGTERM first, SIGKILL after the grace period: Docker's own stop.
            with suppress(_docker_sdk().errors.DockerException):
                container.stop(timeout=30)

    def remove_tree(self, path: str, *, log: Any) -> None:
        # The tree is on the unit's volume, so the unit is what can reach it.
        done = self.run(["rm", "-rf", posixpath.normpath(path)], env={}, log=log)
        if done.code != 0:
            raise AirflowException(f"could not remove {path}: {' | '.join(done.tail)}")


__all__ = [
    "TAIL_LINES",
    "Completed",
    "DockerRunner",
    "DockerUnit",
    "LocalRunner",
    "MbtDocker",
    "Runner",
]
