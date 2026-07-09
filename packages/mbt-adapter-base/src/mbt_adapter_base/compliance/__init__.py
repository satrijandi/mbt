"""The adapter compliance suite (TSD §12.4, FR-ADPT-05).

Passing this suite is the ship bar for a training adapter. Usage::

    from mbt_adapter_base.compliance import TrainingAdapterCompliance

    class TestMyAdapterCompliance(TrainingAdapterCompliance):
        adapter_factory = MyTrainingAdapter
        plugin_module = "my_adapter.plugin"
        framework_modules = ("myframework",)

Requires the ``mbt-adapter-base[compliance]`` extra (pytest + numpy).
"""

from mbt_adapter_base.compliance.suite import (
    PredictionStoreCompliance,
    TrainingAdapterCompliance,
    tiny_binary_dataset,
)

__all__ = ["PredictionStoreCompliance", "TrainingAdapterCompliance", "tiny_binary_dataset"]
