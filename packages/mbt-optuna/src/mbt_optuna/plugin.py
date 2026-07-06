"""The optuna tuning-engine plugin descriptor (import-light, ADR-14)."""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin

from mbt_optuna.engine import OptunaTuningEngine

PLUGIN = AdapterPlugin(
    name="optuna",
    contract_version=CONTRACT_VERSION,
    tuning=OptunaTuningEngine,
    fingerprint_packages=["optuna"],
)
