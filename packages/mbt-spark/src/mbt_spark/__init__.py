"""mbt Spark adapters: lakehouse data, cluster compute, SparkML training."""

from mbt_spark.compute import SparkComputeAdapter
from mbt_spark.data import SparkAdapterError, SparkDataAdapter
from mbt_spark.training import SparkMLTrainingAdapter

__version__ = "0.1.0"

__all__ = [
    "SparkAdapterError",
    "SparkComputeAdapter",
    "SparkDataAdapter",
    "SparkMLTrainingAdapter",
]
