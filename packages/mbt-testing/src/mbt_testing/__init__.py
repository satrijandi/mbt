"""mbt-testing: fake, contract-conformant adapters for tests."""

from mbt_testing.adapters import (
    FakeModel,
    FakeParams,
    FakeRegistryAdapter,
    FakeTrackingAdapter,
    FakeTrainingAdapter,
    FakeTuningEngine,
    InlineComputeAdapter,
)

__version__ = "0.1.0"

__all__ = [
    "FakeModel",
    "FakeParams",
    "FakeRegistryAdapter",
    "FakeTrackingAdapter",
    "FakeTrainingAdapter",
    "FakeTuningEngine",
    "InlineComputeAdapter",
]
