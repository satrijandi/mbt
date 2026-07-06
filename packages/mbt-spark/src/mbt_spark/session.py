"""SparkSession management (lazy pyspark import, ADR-14)."""

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

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
    )
    for key, value in (conf or {}).items():
        builder = builder.config(str(key), str(value))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session
