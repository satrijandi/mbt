"""The sklearn adapter plugin descriptor (import-light, ADR-14)."""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_sklearn.adapter import SklearnTrainingAdapter

PLUGIN = AdapterPlugin(
    name="sklearn",
    contract_version=CONTRACT_VERSION,
    training=SklearnTrainingAdapter,
    fingerprint_packages=["scikit-learn"],
)
