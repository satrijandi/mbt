"""The spark adapter plugin descriptor (import-light, ADR-14).

Importing this module must not import pyspark (or start a JVM); sessions
spin up lazily inside adapter methods.
"""

from mbt_adapter_base import CONTRACT_VERSION, AdapterPlugin
from mbt_spark.compute import SparkComputeAdapter
from mbt_spark.data import SparkDataAdapter
from mbt_spark.training import SparkMLTrainingAdapter

PLUGIN = AdapterPlugin(
    name="spark",
    contract_version=CONTRACT_VERSION,
    data=SparkDataAdapter,
    compute=SparkComputeAdapter,
    training=SparkMLTrainingAdapter,
    fingerprint_packages=["pyspark"],
)
