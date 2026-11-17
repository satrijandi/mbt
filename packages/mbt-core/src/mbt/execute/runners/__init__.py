"""Node runners: datasets in-process, models via ComputeAdapter (TSD §10.4/§10.5).

One module per runner, so each can be read on its own (FEEDBACK v7, "Smaller
notes": ``runners.py`` was 1,530 lines holding the context and four runners):

- ``context``: ``ExecutionContext``, ``BuildContext`` and the node lifecycle
- ``dataset``: ``DatasetRunner``
- ``model``: ``ModelRunner`` (train, judge, register, finish) and ``ModelTestRunner``
- ``scoring``: ``ScoringRunner``
- ``champion_config``: reading a champion's inference config, shared by the last two

This package re-exports the public names, so ``from mbt.execute.runners import
ModelRunner`` is still the way to reach a runner.
"""

from mbt.execute.runners.champion_config import check_hooks_parity, read_inference_config
from mbt.execute.runners.context import (
    BuildContext,
    ExecutionContext,
    evaluation_node_result,
    materialization_key,
    run_with_lifecycle,
)
from mbt.execute.runners.dataset import DatasetRunner
from mbt.execute.runners.model import ModelRunner, ModelTestRunner
from mbt.execute.runners.scoring import ScoringRunner, scoring_run_key
from mbt.quality.judgement import gate_failure_summary

__all__ = [
    "BuildContext",
    "DatasetRunner",
    "ExecutionContext",
    "ModelRunner",
    "ModelTestRunner",
    "ScoringRunner",
    "check_hooks_parity",
    "evaluation_node_result",
    "gate_failure_summary",
    "materialization_key",
    "read_inference_config",
    "run_with_lifecycle",
    "scoring_run_key",
]
