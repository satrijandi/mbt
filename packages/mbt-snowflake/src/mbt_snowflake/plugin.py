"""The snowflake adapter plugin descriptor (import-light, ADR-14).

Importing this module must not import snowflake.connector; connections are
created lazily inside adapter methods so ``mbt parse`` stays fast.
"""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_snowflake.adapter import SnowflakeDataAdapter

PLUGIN = AdapterPlugin(
    name="snowflake",
    contract_version=CONTRACT_VERSION,
    data=SnowflakeDataAdapter,
    fingerprint_packages=["snowflake-connector-python"],
)
