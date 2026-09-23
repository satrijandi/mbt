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
from mbt_testing.data import InMemoryDataAdapter, InMemoryDataError

__version__ = "0.1.0"

__all__ = [
    "FakeModel",
    "FakeParams",
    "FakeRegistryAdapter",
    "FakeTrackingAdapter",
    "FakeTrainingAdapter",
    "FakeTuningEngine",
    "InMemoryDataAdapter",
    "InMemoryDataError",
    "InlineComputeAdapter",
]
