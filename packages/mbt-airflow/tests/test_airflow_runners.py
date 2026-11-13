"""Running mbt in a container, and the operators that do (ADR-34).

The Docker SDK is replaced by a fake that records what the runner asked of it, so
every branch is reachable without a daemon. What the real daemon does with those
calls (a bind mount read back after the container exits, the exit code, stderr and
stdout kept apart) is checked by the showcase's live tier, which runs these DAGs.
"""

import io
import logging
import sys
import tarfile
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from airflow.sdk.exceptions import AirflowException, AirflowFailException
from airflow_helpers import ANCHOR
from mbt_airflow import (
    DockerUnit,
    MbtCleanupOperator,
    MbtCompileOperator,
    MbtDocker,
    MbtNodeOperator,
    TaskSpec,
)
from mbt_airflow.runners import DockerRunner

LOG = logging.getLogger("test")
NODE = TaskSpec(
    "churn_classifier",
    "model.churn_demo.churn_classifier",
    "model",
    "build",
    ("churn_training_set",),
)


class DockerException(Exception):
    """Stands for ``docker.errors.DockerException``."""


class NotFound(DockerException):
    """Stands for ``docker.errors.NotFound``."""


def tar_of(name: str, content: bytes) -> list[bytes]:
    """What ``get_archive`` streams back: a tar holding the one file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    data = buffer.getvalue()
    return [data[:100], data[100:]]  # chunked, as the API streams it


class FakeContainer:
    def __init__(self, world: "FakeDocker") -> None:
        self.world = world

    def wait(self) -> dict[str, int]:
        if self.world.wait_raises:
            raise DockerException("daemon went away")
        return {"StatusCode": self.world.exit_code}

    def logs(self, *, stdout: bool, stderr: bool) -> bytes:
        return (self.world.stdout if stdout else self.world.stderr).encode()

    def get_archive(self, path: str) -> tuple[Iterator[bytes], dict]:
        self.world.reads.append(path)
        if path in self.world.archives:
            return iter(self.world.archives[path]), {}
        if path not in self.world.files:
            raise NotFound(path)
        return iter(tar_of(Path(path).name, self.world.files[path])), {}

    def remove(self, *, force: bool) -> None:
        self.world.removed.append(force)
        if self.world.remove_raises:
            raise DockerException("already gone")

    def stop(self, *, timeout: int) -> None:
        self.world.stopped.append(timeout)
        if self.world.stop_raises:
            raise DockerException("already gone")


class FakeDocker:
    """The SDK's surface the runner uses, and a record of how it was used."""

    def __init__(self) -> None:
        self.exit_code = 0
        self.stdout = ""
        self.stderr = ""
        self.files: dict[str, bytes] = {}
        self.archives: dict[str, list[bytes]] = {}  # raw tar streams, for odd archives
        self.run_raises = False
        self.wait_raises = False
        self.remove_raises = False
        self.stop_raises = False
        self.from_env_timeout: int | None = None
        self.runs: list[dict[str, Any]] = []
        self.reads: list[str] = []
        self.removed: list[bool] = []
        self.stopped: list[int] = []
        self.closed = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeDocker":
        world = self
        errors = types.ModuleType("docker.errors")
        errors.DockerException = DockerException  # type: ignore[attr-defined]
        errors.NotFound = NotFound  # type: ignore[attr-defined]
        sdk = types.ModuleType("docker")
        sdk.errors = errors  # type: ignore[attr-defined]

        def from_env(*, timeout: int) -> Any:
            world.from_env_timeout = timeout
            return types.SimpleNamespace(
                containers=types.SimpleNamespace(run=world.run), close=world.close
            )

        sdk.from_env = from_env  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "docker", sdk)
        monkeypatch.setitem(sys.modules, "docker.errors", errors)
        return self

    def run(self, image: str, command: list[str], **kwargs: Any) -> FakeContainer:
        if self.run_raises:
            raise DockerException("No such image")
        self.runs.append({"image": image, "command": command, **kwargs})
        return FakeContainer(self)

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return FakeDocker().install(monkeypatch)


UNIT = DockerUnit(
    image="registry/churn@sha256:abc",
    network="stack_default",
    volumes={"/host/ws": "/workspace"},
    working_dir="/app/project",
    env={"AWS_REGION": "us-east-1", "WHO": "unit"},
)


def runner(unit: DockerUnit | Any = UNIT, **kwargs: Any) -> DockerRunner:
    return DockerRunner(MbtDocker(unit, **kwargs))


# -- starting the container ---------------------------------------------------------


def test_a_container_gets_the_unit_the_command_and_the_merged_environment(
    docker: FakeDocker,
) -> None:
    done = runner().run(
        ["mbt", "build"], env={"WHO": "task", "MBT_AIRFLOW_TARGET_PATH": "/w/t"}, log=LOG
    )
    assert done.code == 0
    [started] = docker.runs
    assert started == {
        "image": "registry/churn@sha256:abc",
        "command": ["mbt", "build"],
        "detach": True,
        "network": "stack_default",
        "working_dir": "/app/project",
        "volumes": {"/host/ws": {"bind": "/workspace", "mode": "rw"}},
        # a container inherits nothing; the task's own variables win over the unit's
        "environment": {
            "AWS_REGION": "us-east-1",
            "WHO": "task",
            "MBT_AIRFLOW_TARGET_PATH": "/w/t",
        },
    }


def test_the_api_timeout_outlasts_a_slow_create(docker: FakeDocker) -> None:
    """docker-py's own 60 seconds failed tasks before mbt started on a busy daemon."""
    runner().run(["mbt"], env={}, log=LOG)
    assert docker.from_env_timeout == 600
    runner(api_timeout=900).run(["mbt"], env={}, log=LOG)
    assert docker.from_env_timeout == 900
    assert docker.closed


def test_the_unit_is_resolved_when_the_task_runs(docker: FakeDocker) -> None:
    pinned = {"image": "first"}
    config = MbtDocker(lambda: DockerUnit(image=pinned["image"]))
    pinned["image"] = "second"  # the deploy repo moved after the DAG was parsed
    DockerRunner(config).run(["mbt"], env={}, log=LOG)
    assert docker.runs[0]["image"] == "second"


def test_a_unit_that_cannot_be_resolved_fails_the_task_with_its_own_message(
    docker: FakeDocker,
) -> None:
    def no_pin() -> DockerUnit:
        raise AirflowException("images.env has no IMAGE pin yet")

    with pytest.raises(AirflowException, match="no IMAGE pin yet"):
        DockerRunner(MbtDocker(no_pin)).run(["mbt"], env={}, log=LOG)
    assert docker.runs == []


def test_a_container_that_cannot_start_is_a_retryable_error(docker: FakeDocker) -> None:
    docker.run_raises = True
    with pytest.raises(
        AirflowException, match=r"cannot start registry/churn.*No such image"
    ) as info:
        runner().run(["mbt"], env={}, log=LOG)
    assert not isinstance(info.value, AirflowFailException)  # the image may arrive; retry
    assert docker.closed


def test_without_the_docker_sdk_the_task_says_how_to_get_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "docker", None)  # an import of it now fails
    with pytest.raises(AirflowFailException, match=r"pip install 'mbt-airflow\[docker\]'"):
        runner().run(["mbt"], env={}, log=LOG)


# -- reading it back ----------------------------------------------------------------


def test_the_exit_code_the_tail_and_the_files_come_back(docker: FakeDocker) -> None:
    docker.exit_code = 2
    docker.stderr = "one\ntwo\r\n\nthree\nfour\nfive\n"
    docker.files = {"/workspace/t/run_results.json": b'{"results": []}'}
    done = runner().run(
        ["mbt"],
        env={},
        log=LOG,
        collect=["/workspace/t/run_results.json", "/workspace/t/absent.json"],
    )
    assert done.code == 2
    assert done.tail == ("two", "three", "four", "five")  # four lines, none blank
    assert done.files == {"/workspace/t/run_results.json": b'{"results": []}'}
    assert done.stdout == ""  # not asked for
    assert docker.reads == ["/workspace/t/run_results.json", "/workspace/t/absent.json"]


def test_stdout_comes_back_only_when_asked_for(docker: FakeDocker) -> None:
    docker.stdout = "2026-11-12T00:00:00\n"
    assert runner().run(["x"], env={}, log=LOG, capture_stdout=True).stdout == docker.stdout


def test_a_directory_is_no_file(docker: FakeDocker) -> None:
    """A directory is an archive with no regular file in it: nothing to read back."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        directory = tarfile.TarInfo("t")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
    docker.archives = {"/workspace/t": [buffer.getvalue()]}
    assert runner().run(["x"], env={}, log=LOG, collect=["/workspace/t"]).files == {}


def test_the_container_is_removed_even_when_waiting_on_it_fails(docker: FakeDocker) -> None:
    docker.wait_raises = True
    with pytest.raises(DockerException, match="went away"):
        runner().run(["mbt"], env={}, log=LOG)
    assert docker.removed == [True]  # force: it may still be running


def test_a_container_that_is_already_gone_does_not_mask_the_result(docker: FakeDocker) -> None:
    docker.remove_raises = True
    assert runner().run(["mbt"], env={}, log=LOG).code == 0


# -- killing it ---------------------------------------------------------------------


def test_stopping_asks_the_container_to_stop_with_a_grace_period(docker: FakeDocker) -> None:
    unit_runner = runner()
    unit_runner._container = FakeContainer(docker)
    unit_runner.stop()
    assert docker.stopped == [30]  # SIGTERM, then SIGKILL after 30 seconds


def test_stopping_a_container_that_has_exited_is_harmless(docker: FakeDocker) -> None:
    unit_runner = runner()
    unit_runner.stop()  # nothing has started
    docker.stop_raises = True
    unit_runner._container = FakeContainer(docker)
    unit_runner.stop()  # it exited between the check and the call
    assert docker.stopped == [30]


# -- removing a workspace -----------------------------------------------------------


def test_a_workspace_is_removed_by_the_unit_that_can_reach_it(docker: FakeDocker) -> None:
    runner().remove_tree("/workspace/mbt-airflow/dag/run/../run", log=LOG)
    assert docker.runs[0]["command"] == ["rm", "-rf", "/workspace/mbt-airflow/dag/run"]


def test_a_workspace_that_will_not_go_is_an_error(docker: FakeDocker) -> None:
    docker.exit_code = 1
    docker.stderr = "rm: cannot remove: Permission denied"
    with pytest.raises(AirflowException, match=r"could not remove.*Permission denied"):
        runner().remove_tree("/workspace/x", log=LOG)


# -- the operators, in a container --------------------------------------------------

DOCKER = MbtDocker(UNIT)


def node_in_container(**kwargs: Any) -> MbtNodeOperator:
    return MbtNodeOperator(
        task_id=NODE.task_id,
        spec=NODE,
        project_dir="/app/project",
        workspace="/workspace/mbt-airflow/dag/run",
        anchor=ANCHOR,
        docker=DOCKER,
        **kwargs,
    )


def test_a_node_in_a_container_reads_its_result_back_through_the_runner(
    docker: FakeDocker,
) -> None:
    results = '{"results": [{"unique_id": "%s", "status": "success", "metrics": {"auc": 0.7}}]}'
    docker.files = {
        "/workspace/mbt-airflow/dag/run/churn_classifier/run_results.json": (
            (results % NODE.uid).encode()
        )
    }
    outcome = node_in_container().execute({})
    assert outcome == {"node": NODE.uid, "status": "success", "metrics": {"auc": 0.7}}
    [started] = docker.runs
    assert started["command"][:3] == ["mbt", "build", "--project-dir"]
    assert started["environment"]["MBT_AIRFLOW_TARGET_PATH"] == (
        "/workspace/mbt-airflow/dag/run/churn_classifier"
    )
    assert started["environment"]["MBT_AIRFLOW_MANIFEST"] == (
        "/workspace/mbt-airflow/dag/run/mbt_compile/manifest.json"
    )


def test_a_quality_verdict_in_a_container_is_not_retried(docker: FakeDocker) -> None:
    docker.exit_code = 2
    with pytest.raises(AirflowFailException, match="quality verdict"):
        node_in_container().execute({})


def test_a_hard_error_in_a_container_is_retried_and_quotes_the_tail(docker: FakeDocker) -> None:
    docker.exit_code = 1
    docker.stderr = "Error: no such target 'does_not_exist'"
    with pytest.raises(AirflowException, match=r"hard error.*no such target") as info:
        node_in_container().execute({})
    assert not isinstance(info.value, AirflowFailException)


def test_killing_a_task_in_a_container_stops_its_container(docker: FakeDocker) -> None:
    operator = node_in_container()
    operator._runner = unit_runner = DockerRunner(DOCKER)
    unit_runner._container = FakeContainer(docker)
    operator.on_kill()
    assert docker.stopped == [30]


def test_the_compile_task_checks_the_manifest_it_reads_back(
    docker: FakeDocker, churn_demo: tuple[Path, Path]
) -> None:
    path = "/workspace/mbt-airflow/dag/run/mbt_compile/manifest.json"
    operator = MbtCompileOperator(
        task_id="mbt_compile",
        project_dir="/app/project",
        workspace="/workspace/mbt-airflow/dag/run",
        anchor=ANCHOR,
        docker=DOCKER,
        plan_command="build",
        plan_select=["tag:weekly"],
        plan_exclude=None,
        rendered_for=["mbt_compile", "gone_model"],
    )
    docker.files = {path: churn_demo[1].read_bytes()}
    with pytest.raises(AirflowFailException, match="stale"):
        operator.execute({})
    docker.files = {}  # a compile that exited 0 and wrote nothing is not a pass
    with pytest.raises(AirflowException, match="left no manifest"):
        operator.execute({})


def test_cleanup_in_a_container_compares_paths_as_text_and_removes_through_the_unit(
    docker: FakeDocker,
) -> None:
    def cleanup(workspace: str) -> MbtCleanupOperator:
        return MbtCleanupOperator(
            task_id="c", workspace=workspace, workspace_root="/workspace/mbt-airflow", docker=DOCKER
        )

    assert cleanup("/workspace/mbt-airflow/dag/run").execute({}) == "/workspace/mbt-airflow/dag/run"
    assert docker.runs[0]["command"] == ["rm", "-rf", "/workspace/mbt-airflow/dag/run"]
    for escape in ("/workspace/mbt-airflow", "/workspace/mbt-airflow/../x", "/etc"):
        with pytest.raises(AirflowFailException, match="refusing to remove"):
            cleanup(escape).execute({})
    assert len(docker.runs) == 1  # nothing else was removed
