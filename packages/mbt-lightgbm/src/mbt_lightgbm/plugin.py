"""The lightgbm adapter plugin descriptor (import-light, ADR-14)."""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_lightgbm.adapter import LightGBMTrainingAdapter

PLUGIN = AdapterPlugin(
    name="lightgbm",
    contract_version=CONTRACT_VERSION,
    training=LightGBMTrainingAdapter,
    fingerprint_packages=["lightgbm"],
)
