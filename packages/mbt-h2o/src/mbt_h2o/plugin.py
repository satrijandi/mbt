"""The h2o_automl adapter plugin descriptor (import-light, ADR-14).

Importing this module must not import h2o (or start a JVM); clusters spin
up lazily inside train/evaluate/load.
"""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_h2o.adapter import H2OAutoMLAdapter

PLUGIN = AdapterPlugin(
    name="h2o_automl",
    contract_version=CONTRACT_VERSION,
    training=H2OAutoMLAdapter,
    fingerprint_packages=["h2o"],
)
