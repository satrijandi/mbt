"""The evidently report-engine plugin descriptor (import-light, ADR-14)."""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_evidently.engine import EvidentlyReportingEngine

PLUGIN = AdapterPlugin(
    name="evidently",
    contract_version=CONTRACT_VERSION,
    reporting=EvidentlyReportingEngine,
    # Display only: a drift report never changes a model or a gate, so
    # evidently's version stays out of the environment digest (ADR-30).
    fingerprint_packages=[],
)
