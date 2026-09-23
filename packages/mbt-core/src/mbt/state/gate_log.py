"""How many times one test window has arbitrated a gate (D-3).

Every training run evaluates its gates against the test window, and mbt is
built to be run repeatedly. Nothing recorded how many times a given
``(dataset, window)`` pair had judged a candidate, and nothing surfaced it.

Over enough iterations a held-out test window stops being held out. The
decisions are not being made by the model any more, they are being made by the
analyst reading pass/fail and adjusting - which is selection on the test set,
conducted one commit at a time.

**What this is not.** It is not a control. ADR-30's after-test window is the
strong defence and is already present, and `promote` already refuses a version
whose recorded after-test verdict is ``false`` - that enforcement is built and
shipped, and this does not touch it. `feedback-v1.md:1186` (P2) closed the
selection bias *inside* a single run with the robust bootstrap objective and
nested CV; this is the bias *across* runs, which those fixes do not reach.

So this counts and warns, and carries no policy. It is worth doing for the same
reason the backtest std is reported next to the backtest mean: it tells a reader
how much to trust a number that otherwise looks unconditional.

**Where it lives.** ``target/gate_history.json``, beside ``run_results.json``.
That makes it per-checkout and per-clone, which understates the count on a
fresh CI runner - deliberately: a shared, authoritative counter would be state
mbt does not own, and understating is the safe direction for a signal whose
only action is a warning.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Beside ``run_results.json``; ``mbt clean`` removes it with the rest of target/.
GATE_LOG_FILE = "gate_history.json"

#: Evaluations of one (dataset, window) pair past which the window is worth a
#: warning. Twenty candidates judged against one held-out window is enough
#: selection pressure to be worth a reader's attention, and few enough that a
#: normal iterate-a-few-times week never trips it.
REUSE_WARN_THRESHOLD = 20


@dataclass(frozen=True)
class WindowUse:
    """One ``(dataset, test window)`` pair and how often it has judged."""

    dataset_uid: str
    window: tuple[str, str]
    count: int

    @property
    def overused(self) -> bool:
        return self.count >= REUSE_WARN_THRESHOLD


def _log_path(project_dir: Path) -> Path:
    return project_dir / "target" / GATE_LOG_FILE


def _key(dataset_uid: str, window: tuple[str, str]) -> str:
    return f"{dataset_uid}|{window[0]}|{window[1]}"


def read_log(project_dir: Path) -> dict[str, int]:
    """The recorded counts, or an empty log.

    A malformed or unreadable log reads as empty: this is an observability
    signal, and failing a build over its bookkeeping would be worse than
    losing the count.
    """
    path = _log_path(project_dir)
    try:
        loaded: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): int(v) for k, v in loaded.items() if isinstance(v, int | float)}


def record_evaluation(
    project_dir: Path, dataset_uid: str, window: tuple[str, str] | None
) -> WindowUse | None:
    """Count one gate evaluation against ``window``; return the new total.

    None when the node has no resolved test window (a random split, or a node
    that did not resolve one), because there is no window to have reused.
    """
    if window is None:
        return None
    log = read_log(project_dir)
    key = _key(dataset_uid, window)
    count = log.get(key, 0) + 1
    log[key] = count
    path = _log_path(project_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log, indent=1, sort_keys=True) + "\n")
    except OSError:
        pass  # the count is a signal, not a contract; never fail a run for it
    return WindowUse(dataset_uid=dataset_uid, window=window, count=count)


def reuse_warning(use: WindowUse) -> str:
    """What to tell an operator about an over-used window."""
    return (
        f"this test window [{use.window[0]}, {use.window[1]}) has now judged "
        f"{use.count} candidates of {use.dataset_uid} (at or over "
        f"{REUSE_WARN_THRESHOLD}): a window selected against this many times is "
        "no longer fully held out, because the choices between runs were made by "
        "reading its verdicts - treat its metrics as optimistic, and judge the "
        "model on the after-test window (ADR-30) or a fresh test window before "
        "promoting"
    )


__all__ = [
    "GATE_LOG_FILE",
    "REUSE_WARN_THRESHOLD",
    "WindowUse",
    "read_log",
    "record_evaluation",
    "reuse_warning",
]
