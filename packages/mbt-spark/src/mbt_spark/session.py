"""SparkSession management (lazy pyspark import, ADR-14)."""

import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def get_session(
    master: str = "local[*]",
    conf: dict[str, Any] | None = None,
    app_name: str = "mbt",
) -> "SparkSession":
    """Get or create a SparkSession; quiet by default for CLI ergonomics."""
    from pyspark.sql import SparkSession

    conf = dict(conf or {})
    if (
        master.startswith("local")
        and "spark.pyspark.python" not in conf
        and "PYSPARK_PYTHON" not in os.environ
    ):
        # Local executors spawn whatever `python3` is on PATH, which need not
        # be the driver's interpreter (PYTHON_VERSION_MISMATCH when the venv
        # is not activated). Pin executors to the driver unless the caller
        # already did; remote masters keep the image contract (ADR-17).
        conf["spark.pyspark.python"] = sys.executable

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
    )
    for key, value in conf.items():
        builder = builder.config(str(key), str(value))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session
