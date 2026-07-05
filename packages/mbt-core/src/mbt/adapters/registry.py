"""Adapter plugin discovery and resolution (TSD §12.3, FR-ADPT-02).

Plugins are discovered via the ``mbt.adapters`` entry-point group. Loading a
plugin imports only its descriptor module, which by contract is cheap
(no ML framework imports at module level, ADR-14) - that is what keeps
``mbt parse`` inside its 2 s budget (NFR-03).
"""

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from mbt.contracts import CONTRACT_VERSION, AdapterPlugin, TaskType
from mbt.exceptions import ConfigError


def _parse_version(version: str) -> tuple[int, int]:
    try:
        major, minor = version.split(".", 1)
        return int(major), int(minor)
    except ValueError as exc:
        raise ConfigError(
            f"invalid adapter contract version: {version!r}",
            hint="contract versions look like '1.0'",
        ) from exc


@dataclass
class _Entry:
    name: str
    load: Any  # importlib EntryPoint.load
    plugin: AdapterPlugin | None = None


class AdapterRegistry:
    """Lazily loads adapter plugins by name and checks contract compatibility."""

    def __init__(self, core_contract: str = CONTRACT_VERSION) -> None:
        self._core_contract = _parse_version(core_contract)
        self._entries: dict[str, _Entry] = {}
        self._task_schemas_registered: set[str] = set()
        self._discover()

    def _discover(self) -> None:
        for ep in entry_points(group="mbt.adapters"):
            self._entries[ep.name] = _Entry(name=ep.name, load=ep.load)

    def register(self, plugin: AdapterPlugin) -> None:
        """Register a plugin object directly (tests, embedded use)."""
        self._check_contract(plugin)
        self._entries[plugin.name] = _Entry(name=plugin.name, load=lambda: plugin, plugin=plugin)

    @property
    def available(self) -> list[str]:
        return sorted(self._entries)

    def is_installed(self, name: str) -> bool:
        return name in self._entries

    def get(self, name: str) -> AdapterPlugin:
        """Load a plugin by name; a missing adapter names its pip package."""
        entry = self._entries.get(name)
        if entry is None:
            raise ConfigError(
                f"adapter {name!r} is not installed",
                hint=(
                    f"pip install mbt-{name} (installed adapters: "
                    f"{', '.join(self.available) or 'none'})"
                ),
            )
        if entry.plugin is None:
            plugin = entry.load()
            if not isinstance(plugin, AdapterPlugin):
                raise ConfigError(
                    f"entry point for adapter {name!r} did not yield an AdapterPlugin",
                    hint="expose PLUGIN = AdapterPlugin(...) in the plugin module",
                )
            self._check_contract(plugin)
            entry.plugin = plugin
            self._register_task_schemas(plugin)
        return entry.plugin

    def _check_contract(self, plugin: AdapterPlugin) -> None:
        major, minor = _parse_version(plugin.contract_version)
        core_major, core_minor = self._core_contract
        if major != core_major or minor > core_minor:
            raise ConfigError(
                f"adapter {plugin.name!r} pins contract {plugin.contract_version}, "
                f"but this mbt-core supports {core_major}.{core_minor}",
                hint=(
                    f"upgrade {'mbt-core' if major > core_major or minor > core_minor else f'mbt-{plugin.name}'} "
                    "so contract majors match and the adapter's minor is not newer"
                ),
            )

    def _register_task_schemas(self, plugin: AdapterPlugin) -> None:
        if plugin.name in self._task_schemas_registered:
            return
        self._task_schemas_registered.add(plugin.name)
        if plugin.task_schemas:
            from mbt.config.tasks import register_task_schema

            for schema_cls in plugin.task_schemas.values():
                register_task_schema(schema_cls())

    # -- typed helpers -------------------------------------------------------

    def training(self, name: str) -> Any:
        plugin = self.get(name)
        if plugin.training is None:
            raise ConfigError(f"adapter {name!r} provides no training adapter")
        return plugin.training({})

    def supported_tasks(self, name: str) -> set[TaskType]:
        plugin = self.get(name)
        if plugin.training is None:
            return set()
        adapter = plugin.training({})
        return set(adapter.supported_tasks)

    def component(self, kind: str, name: str, config: dict[str, Any]) -> Any:
        """Instantiate one adapter component (data/tracking/registry/compute/tuning)."""
        plugin = self.get(name)
        cls = getattr(plugin, kind, None)
        if cls is None:
            raise ConfigError(
                f"adapter {name!r} provides no {kind} adapter",
                hint=f"check the '{kind}:' entry in profiles.yml",
            )
        return cls(config)

    def fingerprint_packages(self) -> list[str]:
        """Packages every *loaded* plugin wants in the env digest (TSD §8.4)."""
        packages: set[str] = set()
        for entry in self._entries.values():
            if entry.plugin is not None:
                packages.update(entry.plugin.fingerprint_packages)
        return sorted(packages)

    def loaded_plugins(self) -> dict[str, AdapterPlugin]:
        return {n: e.plugin for n, e in self._entries.items() if e.plugin is not None}


_registry: AdapterRegistry | None = None


def get_registry() -> AdapterRegistry:
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def set_registry(registry: AdapterRegistry | None) -> None:
    """Testing hook: replace or reset the process-wide registry."""
    global _registry
    _registry = registry
