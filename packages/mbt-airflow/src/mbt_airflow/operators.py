"""Operators that run mbt for one node of the project (ADR-33).

Each operator is a thin shell over the real CLI: it builds an argument vector,
runs it, streams mbt's events into the Airflow task log, and turns the exit code
into the Airflow failure mode that code means (CLI reference, "Exit codes").
All the planning is ``plan.py``'s.
"""

import json
import os
import shlex
import shutil
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from airflow.sdk import BaseOperator, Context
from airflow.sdk.exceptions import AirflowException, AirflowFailException

from mbt.artifacts.manifest import read_manifest
from mbt_airflow.plan import COMPILE, DATASET_COMMANDS, TaskSpec, plan_tasks

#: Exit code of a quality verdict: a gate, check, test or monitor said no.
EXIT_QUALITY = 2
#: How many trailing stderr lines a hard-error message quotes.
_TAIL_LINES = 4


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
        self._proc: subprocess.Popen[str] | None = None

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

    def target_path(self) -> Path:
        """This task's private target directory (ADR-32): one writer per directory."""
        raise NotImplementedError  # pragma: no cover - subclasses define it

    # -- running it -----------------------------------------------------------------

    def _run(self, argv: list[str]) -> tuple[int, list[str]]:
        """Run mbt, streaming its events into the task log; return (exit code, stderr tail).

        Events go to stderr and stdout is command data (the results table), which the
        task has no use for: the outcome comes from ``run_results.json``.
        """
        self.log.info("$ %s", shlex.join(argv))
        tail: deque[str] = deque(maxlen=_TAIL_LINES)
        env = {**os.environ, **(self.env or {})}
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env
            )
        except FileNotFoundError as exc:
            # Deterministic: retrying cannot make an executable appear.
            raise AirflowFailException(
                f"cannot run {argv[0]!r}: {exc.strerror}. Install mbt on the worker, "
                "or point MbtExecution.mbt_executable at it"
            ) from exc
        self._proc = proc
        assert proc.stderr is not None  # PIPE above
        for raw in proc.stderr:
            line = raw.rstrip()
            if line:
                self.log.info("  %s", line)
                tail.append(line)
        return proc.wait(), list(tail)

    def on_kill(self) -> None:
        """Airflow killed the task (timeout, clear, mark failed): stop mbt too.

        SIGTERM is the signal mbt treats as an interrupt: it terminates its training
        jobs and exits 130 instead of leaving them running.
        """
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def execute(self, context: Context) -> dict[str, Any]:
        code, tail = self._run(self.argv())
        outcome = self.outcome()
        if code == 0:
            self.verify()
            return outcome
        if code == EXIT_QUALITY:
            # A quality verdict is deterministic: retrying cannot change it, so it
            # fails at once and goes to the model owner, not to on-call.
            raise AirflowFailException(f"mbt exited {code} (quality verdict): {outcome}")
        # A hard error has no per-node verdict to read; mbt's own words are the diagnosis.
        raise AirflowException(f"mbt exited {code} (hard error): {' | '.join(tail)}")

    def outcome(self) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover - subclasses define it

    def verify(self) -> None:
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.plan_command = plan_command
        self.plan_select = list(plan_select) if plan_select else None
        self.plan_exclude = list(plan_exclude) if plan_exclude else None
        #: The task ids this DAG was drawn with, to compare against at run time.
        self.rendered_for = list(rendered_for)

    def target_path(self) -> Path:
        return Path(self.workspace) / COMPILE

    def argv(self) -> list[str]:
        argv = [*self._base("compile"), "--anchor", self.anchor]
        argv += ["--target-path", str(self.target_path())]
        return [*argv, "--vars", self.mbt_vars] if self.mbt_vars else argv

    def outcome(self) -> dict[str, Any]:
        return {"manifest": str(self.target_path() / "manifest.json")}

    def verify(self) -> None:
        """Refuse to run a DAG that no longer describes the project (ADR-33).

        The DAG was drawn from a manifest compiled earlier and this run compiled
        again. If the project moved in between (a model gained the selected tag, a
        node was removed) the drawn graph no longer matches what the selector means
        today, and a new node would simply never run behind a green DAG.
        """
        fresh = read_manifest(self.target_path() / "manifest.json", source="run-time manifest")
        current = {
            spec.task_id
            for spec in plan_tasks(fresh, self.plan_command, self.plan_select, self.plan_exclude)
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

    def target_path(self) -> Path:
        return Path(self.workspace) / self.spec.task_id

    def argv(self) -> list[str]:
        workspace = Path(self.workspace)
        argv = [
            *self._base(self.spec.command),
            *["--manifest", str(workspace / COMPILE / "manifest.json")],
            *["--select", self.spec.uid, "--threads", "1"],
            *["--target-path", str(self.target_path())],
        ]
        if self.spec.command in DATASET_COMMANDS:
            # One private state directory per task, ONE dataset cache for the run,
            # and a model that reads its dataset instead of building it (ADR-32).
            argv += ["--no-auto-materialize", "--datasets-path", str(workspace / "datasets")]
        return argv

    def outcome(self) -> dict[str, Any]:
        """This node's own result: unambiguous because the task owns its target directory."""
        path = self.target_path() / "run_results.json"
        if not path.is_file():
            return {"node": self.spec.uid, "status": "no run_results.json"}
        for result in json.loads(path.read_text())["results"]:
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

    def __init__(self, *, workspace: str, workspace_root: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.workspace = workspace
        self.workspace_root = workspace_root

    def execute(self, context: Context) -> str:
        workspace, root = Path(self.workspace).resolve(), Path(self.workspace_root).resolve()
        # Whatever the templates rendered to, never delete outside the workspace root
        # (or the root itself): this task runs rmtree on a path built from a template.
        if workspace == root or not workspace.is_relative_to(root):
            raise AirflowFailException(
                f"refusing to remove {workspace}: it is not inside the workspace root {root}"
            )
        shutil.rmtree(workspace, ignore_errors=True)
        self.log.info("removed %s", workspace)
        return str(workspace)


__all__ = [
    "EXIT_QUALITY",
    "MbtCleanupOperator",
    "MbtCompileOperator",
    "MbtNodeOperator",
]
