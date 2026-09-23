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
    COMPLIANCE_FRACTIONS,
    COMPLIANCE_SAMPLE_KEY,
    COMPLIANCE_SPLIT_SEED,
    CategoricalAdapterCompliance,
    DataAdapterCompliance,
    PredictionStoreCompliance,
    RecordingEvents,
    RegistryAdapterCompliance,
    TrainingAdapterCompliance,
    categorical_dataset,
    tiny_binary_dataset,
    tiny_source_rows,
)

__all__ = [
    "COMPLIANCE_FRACTIONS",
    "COMPLIANCE_SAMPLE_KEY",
    "COMPLIANCE_SPLIT_SEED",
    "CategoricalAdapterCompliance",
    "DataAdapterCompliance",
    "PredictionStoreCompliance",
    "RecordingEvents",
    "RegistryAdapterCompliance",
    "TrainingAdapterCompliance",
    "categorical_dataset",
    "tiny_binary_dataset",
    "tiny_source_rows",
]
