"""The mlflow adapter plugin descriptor (import-light, ADR-14)."""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_mlflow.adapter import MlflowRegistry, MlflowTracking

PLUGIN = AdapterPlugin(
    name="mlflow",
    contract_version=CONTRACT_VERSION,
    tracking=MlflowTracking,
    registry=MlflowRegistry,
    fingerprint_packages=["mlflow"],
)
