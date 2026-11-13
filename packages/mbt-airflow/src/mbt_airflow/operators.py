"""Operators that run mbt for one node of the project (ADR-33).

Each operator is a thin shell over the real CLI: it builds an argument vector,
hands it to a runner (on the worker, or in a container: ``runners.py``), and turns
the exit code into the Airflow failure mode that code means (CLI reference, "Exit
codes"). All the planning is ``plan.py``'s.
"""

import json
import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from airflow.sdk import BaseOperator, Context
from airflow.sdk.exceptions import AirflowException, AirflowFailException

from mbt.artifacts.manifest import read_manifest
from mbt_airflow.plan import COMPILE, DATASET_COMMANDS, TaskSpec, plan_tasks
from mbt_airflow.runners import Completed, DockerRunner, LocalRunner, MbtDocker, Runner

#: Exit code of a quality verdict: a gate, check, test or monitor said no.
EXIT_QUALITY = 2

#: What every mbt process is told about its task, for a wrapper or a hook around
#: ``mbt`` to read (to push the task's metrics, say). They are set last, so
#: ``MbtExecution.env`` cannot shadow them.
ENV_TARGET_PATH = "MBT_AIRFLOW_TARGET_PATH"
ENV_MANIFEST = "MBT_AIRFLOW_MANIFEST"


class _MbtOperator(BaseOperator):
    """What every mbt task shares: where to run, what to run, and how to fail."""

    template_fields: Sequence[str] = (
        "project_dir",
        "workspace",
        "anchor",
        "mbt_vars",
        "mbt_executable",
        "target",
        "profiles_dir",
    )

    def __init__(
        self,
        *,
        project_dir: str,
        workspace: str,
        anchor: str,
        mbt_vars: str = "",
        mbt_executable: str = "mbt",
        target: str | None = None,
        profiles_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        docker: MbtDocker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.project_dir = project_dir
        #: One directory per DAG run (per group): every task's private target
        #: directory is a child of it, and so is the shared dataset cache.
        self.workspace = workspace
        self.anchor = anchor
        self.mbt_vars = mbt_vars
        self.mbt_executable = mbt_executable
        self.target = target
        self.profiles_dir = profiles_dir
        self.env = dict(env) if env else None
        self.docker = docker
        self._runner: Runner | None = None

    # -- what to run ----------------------------------------------------------------

    def _base(self, command: str) -> list[str]:
        argv = [self.mbt_executable, command, "--project-dir", self.project_dir]
        if self.target:
            argv += ["--target", self.target]
        if self.profiles_dir:
            argv += ["--profiles-dir", self.profiles_dir]
        return argv

    def argv(self) -> list[str]:
        raise NotImplementedError  # pragma: no cover - subclasses define it

    def target_path(self) -> str:
        """This task's private target directory (ADR-32): one writer per directory."""
        raise NotImplementedError  # pragma: no cover - subclasses define it

    def collect(self) -> list[str]:
        """The files the outcome is read from, brought back by the runner."""
        raise NotImplementedError  # pragma: no cover - subclasses define it

    def environment(self) -> dict[str, str]:
        """The environment of the mbt process: the configured one, then the task's own."""
        return {**(self.env or {}), ENV_TARGET_PATH: self.target_path()}

    # -- running it -----------------------------------------------------------------

    def make_runner(self) -> Runner:
        return DockerRunner(self.docker) if self.docker else LocalRunner()

    def on_kill(self) -> None:
        """Airflow killed the task (timeout, clear, mark failed): stop mbt too.

        SIGTERM is the signal mbt treats as an interrupt: it terminates its training
        jobs and exits 130 instead of leaving them running.
        """
        if self._runner is not None:
            self._runner.stop()

    def prepare(self, runner: Runner) -> None:
        """Anything the arguments depend on that is only known on the worker."""

    def execute(self, context: Context) -> dict[str, Any]:
        runner = self._runner = self.make_runner()
        self.prepare(runner)
        done = runner.run(self.argv(), env=self.environment(), log=self.log, collect=self.collect())
        outcome = self.outcome(done)
        if done.code in (0, EXIT_QUALITY):
            # The results table mbt prints is stdout, which a task discards; what it
            # holds (status, metrics, the gate's message) is in run_results.json, so
            # the task log gets it from there. A hard error has no result to report.
            self.log.info("result %s", json.dumps(outcome, sort_keys=True))
        if done.code == 0:
            self.verify(done)
            return outcome
        if done.code == EXIT_QUALITY:
            # A quality verdict is deterministic: retrying cannot change it, so it
            # fails at once and goes to the model owner, not to on-call.
            raise AirflowFailException(f"mbt exited {done.code} (quality verdict): {outcome}")
        # A hard error has no per-node verdict to read; mbt's own words are the diagnosis.
        raise AirflowException(f"mbt exited {done.code} (hard error): {' | '.join(done.tail)}")

    def outcome(self, done: Completed) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover - subclasses define it

    def verify(self, done: Completed) -> None:
        """A check that only makes sense after a successful run."""


class MbtCompileOperator(_MbtOperator):
    """The root task: compile the project once, at this run's anchor.

    Every node task then executes THIS manifest verbatim (``--manifest``), so
    snapshots, windows, seeds and cache keys agree across the whole run. ``--vars``
    belong here and nowhere else for the same reason: the manifest carries them.
    """

    def __init__(
        self,
        *,
        plan_command: str,
        plan_select: Sequence[str] | None,
        plan_exclude: Sequence[str] | None,
        rendered_for: Sequence[str],
        plan_phases: bool = True,
        anchor_command: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.plan_command = plan_command
        self.plan_phases = plan_phases
        self.plan_select = list(plan_select) if plan_select else None
        self.plan_exclude = list(plan_exclude) if plan_exclude else None
        #: The task ids this DAG was drawn with, to compare against at run time.
        self.rendered_for = list(rendered_for)
        #: What to run, beside ``mbt``, to learn the anchor when none was pinned.
        self.anchor_command = list(anchor_command) if anchor_command else None

    def target_path(self) -> str:
        return f"{self.workspace}/{COMPILE}"

    def collect(self) -> list[str]:
        return [self.manifest_path()]

    def manifest_path(self) -> str:
        return f"{self.target_path()}/manifest.json"

    def prepare(self, runner: Runner) -> None:
        """Ask the project what time it is, when nothing pinned the anchor.

        A pinned anchor (the ``anchor`` param) always wins. Otherwise a project whose
        clock is its data (a lake that grows a cohort a week) says where it is: the
        command runs where mbt runs, so it sees what mbt sees, and prints the anchor.
        """
        if self.anchor or not self.anchor_command:
            return
        done = runner.run(
            self.anchor_command, env=self.environment(), log=self.log, capture_stdout=True
        )
        if done.code != 0:
            raise AirflowException(
                f"the anchor command exited {done.code}: {' | '.join(done.tail)}"
            )
        lines = done.stdout.split()
        if len(lines) != 1:
            raise AirflowFailException(
                f"the anchor command must print one anchor; it printed {done.stdout!r}"
            )
        self.anchor = lines[0]
        self.log.info("anchor %s", self.anchor)

    def argv(self) -> list[str]:
        argv = [*self._base("compile"), "--anchor", self.anchor]
        argv += ["--target-path", self.target_path()]
        return [*argv, "--vars", self.mbt_vars] if self.mbt_vars else argv

    def outcome(self, done: Completed) -> dict[str, Any]:
        return {"manifest": self.manifest_path()}

    def verify(self, done: Completed) -> None:
        """Refuse to run a DAG that no longer describes the project (ADR-33).

        The DAG was drawn from a manifest compiled earlier and this run compiled
        again. If the project moved in between (a model gained the selected tag, a
        node was removed) the drawn graph no longer matches what the selector means
        today, and a new node would simply never run behind a green DAG.
        """
        raw = done.files.get(self.manifest_path())
        if raw is None:
            raise AirflowException(f"mbt compile left no manifest at {self.manifest_path()}")
        fresh = read_manifest(raw.decode(), source="run-time manifest")
        current = {
            spec.task_id
            for spec in plan_tasks(
                fresh,
                self.plan_command,
                self.plan_select,
                self.plan_exclude,
                phases=self.plan_phases,
            )
        }
        drawn = set(self.rendered_for)
        if current != drawn:
            raise AirflowFailException(
                "this DAG is stale against the project it ran: "
                f"drawn but gone {sorted(drawn - current)}, "
                f"in the project but not drawn {sorted(current - drawn)}. "
                "Recompile the manifest the DAG is drawn from."
            )


class MbtNodeOperator(_MbtOperator):
    """One node of the project, run as its own task with its own target directory."""

    def __init__(self, *, spec: TaskSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.spec = spec

    def target_path(self) -> str:
        """The node's private target directory, shared by all of its phases (ADR-35).

        A model's phases hand their work to each other through a state file there, and
        they never run at the same time, so one directory is one writer at a time.
        """
        return f"{self.workspace}/{self.spec.node}"

    def manifest_path(self) -> str:
        """The run's manifest, compiled once by the root task."""
        return f"{self.workspace}/{COMPILE}/manifest.json"

    def collect(self) -> list[str]:
        return [f"{self.target_path()}/run_results.json"]

    def environment(self) -> dict[str, str]:
        return {**super().environment(), ENV_MANIFEST: self.manifest_path()}

    def argv(self) -> list[str]:
        argv = [
            *self._base(self.spec.command),
            *["--manifest", self.manifest_path()],
            *["--select", self.spec.uid, "--threads", "1"],
            *["--target-path", self.target_path()],
        ]
        if self.spec.phase is not None:
            argv += ["--phase", self.spec.phase]
        if self.spec.command in DATASET_COMMANDS:
            # One private state directory per task, ONE dataset cache for the run,
            # and a model that reads its dataset instead of building it (ADR-32).
            argv += ["--no-auto-materialize", "--datasets-path", f"{self.workspace}/datasets"]
        return argv

    def outcome(self, done: Completed) -> dict[str, Any]:
        """This node's own result: unambiguous because the task owns its target directory."""
        raw = done.files.get(self.collect()[0])
        if raw is None:
            return {"node": self.spec.uid, "status": "no run_results.json"}
        for result in json.loads(raw)["results"]:
            if result["unique_id"] == self.spec.uid:
                kept = {k: result[k] for k in ("status", "message", "metrics") if result.get(k)}
                return {"node": self.spec.uid, **kept}
        return {"node": self.spec.uid, "status": "missing from run_results.json"}


class MbtCleanupOperator(BaseOperator):
    """Remove the run's workspace once every node has succeeded.

    It runs only when everything upstream succeeded (the default trigger rule), so
    a failed run keeps its target directories, logs and results for debugging.
    """

    template_fields: Sequence[str] = ("workspace", "workspace_root")

    def __init__(
        self,
        *,
        workspace: str,
        workspace_root: str,
        docker: MbtDocker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.workspace = workspace
        self.workspace_root = workspace_root
        self.docker = docker

    def execute(self, context: Context) -> str:
        # A container's paths mean nothing on the worker, so they are compared as text;
        # local ones are resolved, so a symlink cannot lead out of the root.
        workspace: PurePath
        root: PurePath
        if self.docker:
            workspace = PurePosixPath(posixpath.normpath(self.workspace))
            root = PurePosixPath(posixpath.normpath(self.workspace_root))
        else:
            workspace, root = Path(self.workspace).resolve(), Path(self.workspace_root).resolve()
        # Whatever the templates rendered to, never delete outside the workspace root
        # (or the root itself): this task removes a path built from a template.
        if workspace == root or not workspace.is_relative_to(root):
            raise AirflowFailException(
                f"refusing to remove {workspace}: it is not inside the workspace root {root}"
            )
        runner = DockerRunner(self.docker) if self.docker else LocalRunner()
        runner.remove_tree(str(workspace), log=self.log)
        return str(workspace)


__all__ = [
    "ENV_MANIFEST",
    "ENV_TARGET_PATH",
    "EXIT_QUALITY",
    "MbtCleanupOperator",
    "MbtCompileOperator",
    "MbtNodeOperator",
]
