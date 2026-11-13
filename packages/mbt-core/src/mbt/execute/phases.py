"""Phases of a node: how ``mbt run --phase`` splits one node's work (ADR-35).

A dataset is built and then checked. A model is trained, judged against its gates,
and registered. Run whole, a node does all of it in one invocation, exactly as it
always has. An orchestrator that wants each step as a task of its own (its own
retry, its own resources, its own place in the graph) asks for one phase at a time,
and the phases hand their work to each other through two things that already exist:

- a dataset's phases share the dataset cache (ADR-32), which ``materialize`` writes
  and ``checks`` reads;
- a model's phases share a small state file under the node's ``--target-path``, which
  ``train`` writes, ``gate`` extends with its judgement, and ``register`` completes.

So the phases of one node belong to ONE target path. They never run at the same time
(each waits for the one before it), which is what the target lock guarantees.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mbt.artifacts.run_results import GateResult, MonitorResult, RegistrationResult
from mbt.exceptions import ConfigError, StateError
from mbt_adapter_base import JobResult, ManifestNode

if TYPE_CHECKING:
    from mbt.artifacts.manifest import Manifest

DATASET_PHASES = ("materialize", "checks")
MODEL_PHASES = ("train", "gate", "register")
#: The phases of each resource kind, in the order they run.
PHASES_BY_KIND: dict[str, tuple[str, ...]] = {"dataset": DATASET_PHASES, "model": MODEL_PHASES}
#: The commands a phase can be asked of: the ones that build and train.
PHASE_COMMANDS = ("run", "build")
STATE_VERSION = 1


def check_phase(command: str, phase: str, manifest: "Manifest", order: Sequence[str]) -> None:
    """Refuse a phase that the command or a selected node cannot honour, before any work.

    Every selected node must have the phase: ``--phase train`` over a selection that
    includes a dataset has no meaning for the dataset, and silently skipping it would
    let an orchestrator believe a dataset was built.
    """
    if command not in PHASE_COMMANDS:
        raise ConfigError(
            f"--phase applies to {' and '.join(repr(c) for c in PHASE_COMMANDS)}, not {command!r}",
            hint="a phase splits the work of building a dataset or training a model; "
            "test, score and monitor have one step per node",
        )
    known = [p for phases in PHASES_BY_KIND.values() for p in phases]
    if phase not in known:
        raise ConfigError(
            f"unknown phase {phase!r}",
            hint="datasets have "
            + ", ".join(DATASET_PHASES)
            + "; models have "
            + ", ".join(MODEL_PHASES),
        )
    lacking = sorted(
        manifest.nodes[uid].name
        for uid in order
        if phase not in PHASES_BY_KIND[manifest.nodes[uid].resource_type]
    )
    if lacking:
        kind = "dataset" if phase in DATASET_PHASES else "model"
        raise ConfigError(
            f"phase {phase!r} is a {kind} phase, but the selection includes {lacking}",
            hint=f"select only {kind}s, for example --select <name> for one node; "
            "a phase runs exactly the nodes selected",
        )


class ModelPhaseState(BaseModel):
    """What a model's phases hand to one another (ADR-35).

    ``train`` writes the job's result, the champion it was compared against, and
    nothing else. ``gate`` adds its judgement. ``register`` adds the version, first as
    ``registered_version`` the moment the registry has it, and then ``registration``
    once the stage is set, so a retry of ``register`` finishes the job instead of
    registering the same artifact twice.
    """

    model_config = ConfigDict(extra="forbid")

    state_version: int = STATE_VERSION
    unique_id: str
    #: The manifest the phases ran against: a state is only valid for its own plan.
    manifest_hash: str
    job_result: JobResult
    #: The champion the job compared against, so the gate labels the comparison it saw.
    champion_version: str | None = None
    #: Seconds spent by the phases that have run, so the node's result can report what
    #: the node cost so far and not only what its last step did.
    elapsed_s: float = 0.0

    gates: list[GateResult] | None = None
    stability: list[MonitorResult] = Field(default_factory=list)
    verdict: str | None = None
    passed: bool | None = None
    failure_summary: str | None = None

    registered_version: str | None = None
    registration: RegistrationResult | None = None


def write_state(path: Path, state: ModelPhaseState) -> None:
    """Write atomically: a phase killed mid-write must not leave half a state behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(state.model_dump_json(indent=2))
    os.replace(partial, path)


def read_state(
    path: Path, node: ManifestNode, manifest_hash: str, *, phase: str
) -> ModelPhaseState:
    """The state an earlier phase left for ``node``, or an error that says what to run."""
    if not path.is_file():
        raise StateError(
            f"nothing to {phase} for model {node.name!r}: no training state at {path}",
            resource=node.unique_id,
            hint="run 'mbt run --phase train' for this model first, with the same "
            "--target-path (a model's phases share one)",
        )
    try:
        state = ModelPhaseState.model_validate_json(path.read_text())
    except ValidationError as exc:
        raise StateError(
            f"the training state at {path} is not readable: {exc.errors()[0]['msg']}",
            resource=node.unique_id,
            hint="run 'mbt run --phase train' again; the state is rewritten from scratch",
        ) from exc
    if state.state_version != STATE_VERSION:
        raise StateError(
            f"the training state at {path} is version {state.state_version}; "
            f"this mbt reads version {STATE_VERSION}",
            resource=node.unique_id,
            hint="run 'mbt run --phase train' again with this mbt",
        )
    if state.unique_id != node.unique_id or state.manifest_hash != manifest_hash:
        raise StateError(
            f"the training state at {path} was left by a different plan "
            f"({state.unique_id} at manifest {state.manifest_hash[:12]})",
            resource=node.unique_id,
            hint="every phase of a run must use the same --manifest; run "
            "'mbt run --phase train' again against this one",
        )
    return state


__all__ = [
    "DATASET_PHASES",
    "MODEL_PHASES",
    "PHASES_BY_KIND",
    "PHASE_COMMANDS",
    "STATE_VERSION",
    "ModelPhaseState",
    "check_phase",
    "read_state",
    "write_state",
]
