"""The context one training job runs against (B-2).

``_JobRuntime`` lived in ``job.py`` as a private, and ``training_report.py``
imported it across a module boundary under ``TYPE_CHECKING`` - a module
reaching into another module's privates for the type of its own first
parameter, which is the signal that the boundary is a file split rather than a
seam.

Hoisting it here makes the dependency honest and one-directional: both
``job.py`` and ``training_report.py`` depend on this, and neither depends on
the other for its types. ``job.py`` can then import ``training_report`` at
module scope, which removes four function-local imports that existed purely to
dodge the cycle - lazy imports are idiomatic here (ADR-14), which is precisely
why that cycle was invisible.
"""

from dataclasses import dataclass
from typing import Any

from mbt.execute.handles import TransformedDatasetHandle
from mbt.quality.hooks import ModelHooks
from mbt_adapter_base import (
    DatasetProfile,
    MetricSpec,
    ModelSpec,
    RunContext,
    TrainingJob,
)


@dataclass
class JobRuntime:
    job: TrainingJob
    spec: ModelSpec
    adapter: Any
    handle: Any  # what the adapter reads (transformed, or a path materialization)
    transformed: TransformedDatasetHandle  # always the lazy transformed view
    base_handle: Any  # pre-transform training view (carries the time_column)
    #: The whole materialization, after-test split included (ADR-30). Only the
    #: training report reads it; everything else goes through ``base_handle``.
    materialization: Any
    base_profile: DatasetProfile
    hooks: ModelHooks | None
    builtin_specs: list[MetricSpec]
    hook_specs: list[MetricSpec]
    ctx: RunContext
    store: Any


__all__ = ["JobRuntime"]
