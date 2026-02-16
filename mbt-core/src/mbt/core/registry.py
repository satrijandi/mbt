"""Plugin registry for discovering and loading MBT adapters via entry_points."""

from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable


class MissingAdapterError(Exception):
    """Raised when a required adapter is not installed."""

    def __init__(self, group: str, name: str):
        self.group = group
        self.name = name
        package_name = f"mbt-{name.replace('_', '-')}"
        super().__init__(
            f"No adapter '{name}' found in group '{group}'.\n"
            f"Install it with: pip install {package_name}"
        )


@runtime_checkable
class Plugin(Protocol):
    """Base protocol for all plugins."""

    pass


class PluginRegistry:
    """Registry for discovering and instantiating MBT plugins.

    Plugins are discovered via Python entry_points mechanism. Each adapter
    package declares its plugins in pyproject.toml:

    [project.entry-points."mbt.frameworks"]
    sklearn = "mbt_sklearn.framework:SklearnFramework"
    h2o_automl = "mbt_h2o.framework:H2OAutoMLFramework"

    [project.entry-points."mbt.model_registries"]
    mlflow = "mbt_mlflow.registry:MLflowRegistry"
    """

    def __init__(self):
        self._cache: dict[tuple[str, str], Any] = {}

    def get(self, group: str, name: str) -> Any:
        """Load and instantiate a plugin by group and name.

        Args:
            group: Plugin group (e.g., "mbt.frameworks", "mbt.model_registries")
            name: Plugin name (e.g., "sklearn", "h2o_automl", "mlflow")

        Returns:
            Instantiated plugin class

        Raises:
            MissingAdapterError: If plugin not found

        Examples:
            >>> registry = PluginRegistry()
            >>> framework = registry.get("mbt.frameworks", "sklearn")
            >>> model_registry = registry.get("mbt.model_registries", "mlflow")
        """
        # Check cache first
        cache_key = (group, name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Discover entry points for this group
        eps = entry_points()

        # Handle both old and new importlib.metadata APIs
        if hasattr(eps, 'select'):
            # New API (Python 3.10+)
            group_eps = eps.select(group=group)
        else:
            # Old API (Python 3.9)
            group_eps = eps.get(group, [])

        # Find the specific plugin
        plugin_ep = None
        for ep in group_eps:
            if ep.name == name:
                plugin_ep = ep
                break

        if plugin_ep is None:
            raise MissingAdapterError(group, name)

        # Load and instantiate
        plugin_class = plugin_ep.load()
        plugin_instance = plugin_class()

        # Cache for future use
        self._cache[cache_key] = plugin_instance

        return plugin_instance

    def list_installed(self) -> dict[str, list[str]]:
        """List all installed plugins by group.

        Returns:
            Dictionary mapping group names to lists of plugin names

        Examples:
            >>> registry = PluginRegistry()
            >>> plugins = registry.list_installed()
            >>> print(plugins)
            {
                'mbt.frameworks': ['sklearn', 'h2o_automl'],
                'mbt.model_registries': ['mlflow'],
                'mbt.storage': ['local', 's3'],
                'mbt.data_connectors': ['local_file', 'snowflake'],
            }
        """
        result: dict[str, list[str]] = {}
        eps = entry_points()

        # Define known MBT plugin groups
        mbt_groups = [
            "mbt.frameworks",
            "mbt.model_registries",
            "mbt.storage",
            "mbt.data_connectors",
            "mbt.executors",
            "mbt.secrets",
            "mbt.orchestrators",
        ]

        for group in mbt_groups:
            # Handle both old and new APIs
            if hasattr(eps, 'select'):
                group_eps = eps.select(group=group)
            else:
                group_eps = eps.get(group, [])

            plugin_names = sorted([ep.name for ep in group_eps])
            if plugin_names:
                result[group] = plugin_names

        return result

    def list_group(self, group: str) -> list[str]:
        """List all plugins in a specific group.

        Args:
            group: Plugin group (e.g., "mbt.frameworks")

        Returns:
            List of plugin names in the group

        Examples:
            >>> registry = PluginRegistry()
            >>> frameworks = registry.list_group("mbt.frameworks")
            >>> print(frameworks)
            ['sklearn', 'h2o_automl']
        """
        eps = entry_points()

        # Handle both old and new APIs
        if hasattr(eps, 'select'):
            group_eps = eps.select(group=group)
        else:
            group_eps = eps.get(group, [])

        return sorted([ep.name for ep in group_eps])

    def has_plugin(self, group: str, name: str) -> bool:
        """Check if a plugin is installed without loading it.

        Args:
            group: Plugin group
            name: Plugin name

        Returns:
            True if plugin is installed, False otherwise
        """
        eps = entry_points()

        if hasattr(eps, 'select'):
            group_eps = eps.select(group=group)
        else:
            group_eps = eps.get(group, [])

        for ep in group_eps:
            if ep.name == name:
                return True

        return False

    def clear_cache(self):
        """Clear the plugin instance cache.

        Useful for testing or when plugins need to be reloaded.
        """
        self._cache.clear()
