"""The built-in ``local`` adapter plugin descriptor (ADR-2)."""

from mbt.adapters.local.compute import LocalComputeAdapter
from mbt.adapters.local.data import LocalDataAdapter
from mbt.contracts import CONTRACT_VERSION, AdapterPlugin

PLUGIN = AdapterPlugin(
    name="local",
    contract_version=CONTRACT_VERSION,
    data=LocalDataAdapter,
    compute=LocalComputeAdapter,
    fingerprint_packages=["duckdb", "pyarrow"],
)
