"""The ``fake`` adapter plugin descriptor (import-light, ADR-14)."""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_testing.adapters import (
    FakeRegistryAdapter,
    FakeReportingEngine,
    FakeTrackingAdapter,
    FakeTrainingAdapter,
    FakeTuningEngine,
    InlineComputeAdapter,
)

PLUGIN = AdapterPlugin(
    name="fake",
    contract_version=CONTRACT_VERSION,
    training=FakeTrainingAdapter,
    tracking=FakeTrackingAdapter,
    registry=FakeRegistryAdapter,
    tuning=FakeTuningEngine,
    reporting=FakeReportingEngine,
    compute=InlineComputeAdapter,
    fingerprint_packages=[],
)
