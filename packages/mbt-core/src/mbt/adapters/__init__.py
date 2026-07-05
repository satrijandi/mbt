"""Adapter discovery and the built-in local adapters."""

from mbt.adapters.registry import (
    AdapterRegistry,
    get_registry,
    set_registry,
)

__all__ = ["AdapterRegistry", "get_registry", "set_registry"]
