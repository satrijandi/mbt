"""Compatibility re-export of the adapter contract surface.

The contracts live in ``mbt-adapter-base`` (versioned independently,
TSD §2, FR-ADPT-01); ``mbt.contracts`` re-exports them so core code and
early adapters can keep importing from one place.
"""

from mbt_adapter_base import *  # noqa: F403
from mbt_adapter_base import __all__ as _base_all
from mbt_adapter_base.protocols import (
    DataBuildContext,
    PythonDataTest,
    SourceTableLike,
)

__all__ = [*_base_all, "DataBuildContext", "PythonDataTest", "SourceTableLike"]
