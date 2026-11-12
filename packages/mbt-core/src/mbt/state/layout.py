"""Where one invocation writes its build artifacts (ADR-32).

``project_dir / "target" / ...`` used to be spelled out at about twenty sites:
the manifest, ``run_results.json``, the run logs, the dataset cache, the scoring
inputs, the gate history, the generated docs and the lock. That was fine while
"one project, one ``target/``" was the only shape. An orchestrator that runs one
node per task needs the opposite: each task its own writable artifacts (the
lock is one writer per target, ``run_results.json`` is one file) over ONE
dataset cache they all share. So the answer to "where does X live" is now this
module's, and the CLI's ``--target-path`` / ``--datasets-path`` are its only
inputs.

Two roots, because the two halves of ``target/`` have different sharing rules:

- ``root`` holds per-invocation STATE: the manifest, results, logs, the gate
  history, the lock. Two writers in one root race, so there is one writer per
  root (``mbt.state.lock``).
- ``datasets`` holds the dataset cache. A materialization is addressed by a
  hash of its pinned inputs and windows, so two invocations that compute the
  same key compute the same bytes: it is safe to share across roots, and it is
  the hand-off between a dataset node and the model nodes that read it.

``datasets`` defaults to ``<root>/datasets``, so a bare ``--target-path`` keeps
the one-directory behaviour and nothing is shared unless asked.
"""

from dataclasses import dataclass
from pathlib import Path

from mbt.exceptions import ConfigError

TARGET_DIRNAME = "target"
DATASETS_DIRNAME = "datasets"


@dataclass(frozen=True)
class TargetLayout:
    """The directories one invocation reads and writes. Build with :meth:`for_project`."""

    #: Per-invocation state: manifest, results, logs, lock, generated docs.
    root: Path
    #: The content-addressed dataset cache; shareable across roots.
    datasets: Path
    #: True when ``root`` was supplied rather than defaulted to ``<project>/target``.
    #: ``mbt clean`` deletes a defaulted root freely and a supplied one only if
    #: mbt demonstrably owns it.
    custom_root: bool = False

    @classmethod
    def for_project(
        cls,
        project_dir: Path,
        target_path: str | Path | None = None,
        datasets_path: str | Path | None = None,
    ) -> "TargetLayout":
        """Resolve the layout for ``project_dir`` and the optional overrides.

        A relative override is taken relative to the project, like every other
        config path; the CLI has already absolutized what the user typed.
        """
        project = project_dir.resolve()
        root = _local_dir(project, target_path, "--target-path") or project / TARGET_DIRNAME
        if project.is_relative_to(root):
            raise ConfigError(
                f"--target-path {root} contains the project directory {project}",
                hint="point it at a directory of its own: mbt deletes and rewrites what is "
                "inside it, and 'mbt clean' would remove the project",
            )
        datasets = _local_dir(project, datasets_path, "--datasets-path") or root / DATASETS_DIRNAME
        return cls(root=root, datasets=datasets, custom_root=target_path is not None)

    # -- the well-known paths ----------------------------------------------------

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def run_results(self) -> Path:
        return self.root / "run_results.json"

    @property
    def gate_history(self) -> Path:
        return self.root / "gate_history.json"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def json_schemas(self) -> Path:
        return self.root / "json-schemas"

    def run_logs(self, run_id: str) -> Path:
        return self.root / "run_logs" / run_id

    def dataset_dir(self, name: str, key: str) -> Path:
        """One dataset materialization: ``<datasets>/<name>/<key>``."""
        return self.datasets / name / key

    def scoring_input_dir(self, name: str, key: str) -> Path:
        """A scoring batch. Never shared and never reused: it is rebuilt each call (F4)."""
        return self.root / "scoring_inputs" / name / key


def _local_dir(project: Path, value: str | Path | None, flag: str) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if "://" in text:
        raise ConfigError(
            f"{flag} must be a local directory, got {text!r}",
            hint="use a path on a disk the tasks share (a mounted volume works); "
            "object-store locations are not supported here",
        )
    return (project / text).resolve()


__all__ = ["DATASETS_DIRNAME", "TARGET_DIRNAME", "TargetLayout"]
