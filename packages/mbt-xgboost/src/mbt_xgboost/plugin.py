"""The xgboost adapter plugin descriptor (import-light, ADR-14).

Importing this module must not import xgboost; the compliance suite
enforces it (TSD §12.4).
"""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin

from mbt_xgboost.adapter import XGBoostTrainingAdapter

PLUGIN = AdapterPlugin(
    name="xgboost",
    contract_version=CONTRACT_VERSION,
    training=XGBoostTrainingAdapter,
    fingerprint_packages=["xgboost"],
)
