"""The typed event vocabulary, shared by core and by every adapter (B-4).

``EventSink.emit`` used to take ``object``, and the bus wrapped anything that
was not an ``Event`` in a bare ``LogMessage`` at the default level. That guess
reached production: the local data adapter reported an empty after-test split
as a typed warning while Spark and Snowflake emitted a bare string, so the same
condition on the same project was a WARN or an informational line depending
only on which data adapter ran (v5 live defect 1). Severity belongs to the
event, not to the adapter that happened to raise it.

The base lives here rather than in ``mbt.events`` because an adapter package
cannot import core. Core re-exports ``Event`` and ``LogMessage`` from
``mbt.events.models`` and adds the events only it emits, so ``isinstance(x,
Event)`` means the same thing on both sides of the seam.

Events named here are the ones an *adapter* emits. Each carries its facts as
fields, so a sink filtering for one matches on its type rather than on its
English - ``mbt monitor`` suppressing the label read's row count used to test
``"to score" in message``, which reworded away.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mbt_adapter_base.types import OUT_OF_TIME_SPLIT

#: Severity, in the order sinks filter on.
Level = Literal["debug", "info", "warn", "error"]


def _now() -> datetime:
    return datetime.now(tz=UTC)


class Event(BaseModel):
    """Base event: name, level, timestamps, correlation ids."""

    model_config = ConfigDict(extra="forbid")

    event: str = ""
    level: Level = "info"
    ts: datetime = Field(default_factory=_now)
    run_id: str | None = None
    unique_id: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.event:
            self.event = type(self).__name__

    def human(self) -> str:
        """One-line human rendering; sinks may add color."""
        return self.event


class LogMessage(Event):
    """Free-form informational message.

    Still the right event for one-off lines with no consumer but a human. It is
    the *fallback* that was the problem, not the type.
    """

    message: str = ""

    def human(self) -> str:
        return self.message


class DatasetMaterialized(Event):
    """One dataset build's per-split row counts, from whichever adapter ran.

    Emitted once by the shared build recipe (``materialization.py``), so the
    three data adapters no longer each decide how to word it - the warehouse
    adapters used to prefix the node id and the local one did not.
    """

    dataset: str = ""
    row_counts: dict[str, int] = Field(default_factory=dict)

    def human(self) -> str:
        detail = ", ".join(f"{split}={count}" for split, count in sorted(self.row_counts.items()))
        return f"materialized {sum(self.row_counts.values())} rows: {detail}"


class EmptyAfterTestSplit(Event):
    """The after-test split materialized zero rows (ADR-30).

    The one empty split that is not an error: a window ending at the anchor is
    routinely empty until newer rows land upstream. It is a WARN because the
    training report then has no after-test period to show and after-test gates
    have nothing to judge - and it is a warn *here*, once, rather than in each
    adapter's own words.
    """

    level: Level = "warn"
    split: str = OUT_OF_TIME_SPLIT
    window: tuple[str, str] | None = None

    def human(self) -> str:
        where = f" [{self.window[0]}, {self.window[1]})" if self.window else ""
        return (
            f"split {self.split!r} materialized 0 rows{where}: the training report "
            "has no after-test period to show, and after-test gates have nothing to judge"
        )


class ScoringInputMaterialized(Event):
    """One unlabeled scoring batch's row count (ADR-20).

    Zero rows is a warning rather than an error - an empty nightly batch is
    legitimate, unlike an empty training split.

    ``mbt monitor`` reads arriving labels through this same build path, so this
    event announces a table that nothing is scoring. The monitor suppresses it
    by type and re-emits the count in its own words; that used to be a substring
    match on the message text.
    """

    rows: int = 0

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        if self.rows == 0 and self.level == "info":
            self.level = "warn"

    def human(self) -> str:
        if self.rows == 0:
            return "scoring input materialized 0 rows; nothing to score"
        return f"scoring input materialized {self.rows} rows to score"


class AdapterMessage(Event):
    """A note from a named adapter that has no richer type yet.

    Distinct from ``LogMessage`` only in carrying the adapter's name as a
    field, so the text does not have to.
    """

    adapter: str = ""
    message: str = ""

    def human(self) -> str:
        return f"{self.adapter}: {self.message}" if self.adapter else self.message


class EarlyStoppingWithoutValidation(Event):
    """``early_stopping_rounds`` is set but the final fit has no validation split.

    Core carves an implicit validation slice for TUNING trials and then
    reabsorbs it for the final fit (ADR-8, deliberately - those rows are
    training data). So every trial that voted on the hyperparameters stopped
    early, and the model that ships trains all ``n_estimators`` rounds.

    WARN, and worded as the consequence rather than as a missing declaration:
    it used to read "declare split.validation on the dataset to stop early",
    which sounds like a feature the user has not switched on rather than
    "the model about to be registered has a different effective complexity
    from the one your search scored" (D-2).
    """

    level: Level = "warn"
    adapter: str = ""
    rounds: int = 0

    def human(self) -> str:
        return (
            f"{self.adapter}: early_stopping_rounds={self.rounds} has no validation split "
            "on the final fit, so it trains every round while the tuning trials that "
            "chose these hyperparameters stopped early - the registered model is "
            "regularized differently from the one the search scored; declare "
            "split.validation on the dataset to hold a slice out of the final fit too"
        )


def as_event(value: object) -> Event:
    """Coerce a foreign object into an ``Event``.

    The ONLY sanctioned use is the hook boundary, where a user's ``hooks.py``
    genuinely may hand the bus anything. Everywhere else the seam is typed, so
    reaching for this means an event type is missing rather than that a string
    needs wrapping.
    """
    return value if isinstance(value, Event) else LogMessage(message=str(value))


__all__ = [
    "AdapterMessage",
    "DatasetMaterialized",
    "EarlyStoppingWithoutValidation",
    "EmptyAfterTestSplit",
    "Event",
    "Level",
    "LogMessage",
    "ScoringInputMaterialized",
    "as_event",
]
