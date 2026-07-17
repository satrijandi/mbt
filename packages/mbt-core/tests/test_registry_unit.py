"""Adapter registry: contract checks, lazy loading, typed helpers (TSD §12.3)."""

from typing import Any, ClassVar

import pytest

from mbt.adapters.registry import AdapterRegistry, _Entry, get_registry, set_registry
from mbt.contracts import CONTRACT_VERSION, AdapterPlugin, TaskType
from mbt.exceptions import ConfigError


class _NullTraining:
    name = "null"
    supported_tasks: ClassVar[set[TaskType]] = {TaskType.BINARY_CLASSIFICATION}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}


class _RegressionSchema:
    task = TaskType.REGRESSION
    allowed_metrics: ClassVar[set[str]] = {"rmse"}

    def validate_spec(self, spec: Any) -> list[Any]:
        return []

    def validate_dataset(self, spec: Any, profile: Any) -> list[Any]:
        return []


def _plugin(name: str = "nulladapter", **overrides: Any) -> AdapterPlugin:
    defaults: dict[str, Any] = {
        "name": name,
        "contract_version": CONTRACT_VERSION,
        "training": _NullTraining,
    }
    defaults.update(overrides)
    return AdapterPlugin(**defaults)


def test_invalid_contract_version_text_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="invalid adapter contract version"):
        AdapterRegistry(core_contract="banana")


def test_register_makes_a_plugin_available(fake_registry: AdapterRegistry) -> None:
    plugin = _plugin()
    fake_registry.register(plugin)
    assert fake_registry.is_installed("nulladapter")
    assert not fake_registry.is_installed("ghost")
    assert "nulladapter" in fake_registry.available
    assert fake_registry.get("nulladapter") is plugin


def test_missing_adapter_names_its_pip_package(fake_registry: AdapterRegistry) -> None:
    with pytest.raises(ConfigError, match="pip install mbt-ghost"):
        fake_registry.get("ghost")


def test_entry_point_yielding_a_non_plugin_is_rejected(fake_registry: AdapterRegistry) -> None:
    not_a_plugin = object()
    fake_registry._entries["broken"] = _Entry(name="broken", load=lambda: not_a_plugin)
    with pytest.raises(ConfigError, match="did not yield an AdapterPlugin"):
        fake_registry.get("broken")


def test_newer_contract_major_is_rejected_with_upgrade_hint(
    fake_registry: AdapterRegistry,
) -> None:
    with pytest.raises(ConfigError, match="upgrade mbt-core") as excinfo:
        fake_registry.register(_plugin(contract_version="99.0"))
    assert "pins contract 99.0" in str(excinfo.value)


def test_older_contract_major_points_at_the_adapter_package(
    fake_registry: AdapterRegistry,
) -> None:
    with pytest.raises(ConfigError, match="upgrade mbt-nulladapter"):
        fake_registry.register(_plugin(contract_version="0.0"))


def test_lazy_load_registers_task_schemas_once(
    fake_registry: AdapterRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mbt.config import tasks as tasks_module

    # register into a scratch copy so the process-wide schema registry survives
    monkeypatch.setattr(tasks_module, "_REGISTRY", dict(tasks_module._REGISTRY))
    plugin = _plugin(task_schemas={TaskType.REGRESSION: _RegressionSchema})
    fake_registry._entries["nulladapter"] = _Entry(name="nulladapter", load=lambda: plugin)

    assert fake_registry.get("nulladapter") is plugin
    assert TaskType.REGRESSION in tasks_module._REGISTRY
    # a second registration attempt is a no-op (would raise "already registered")
    fake_registry._register_task_schemas(plugin)


def test_training_helper_requires_a_training_adapter(fake_registry: AdapterRegistry) -> None:
    fake_registry.register(_plugin(name="dataonly", training=None))
    with pytest.raises(ConfigError, match="provides no training adapter"):
        fake_registry.training("dataonly")
    assert fake_registry.training("fake").name == "fake"


def test_supported_tasks_is_empty_without_a_training_adapter(
    fake_registry: AdapterRegistry,
) -> None:
    fake_registry.register(_plugin(name="dataonly", training=None))
    assert fake_registry.supported_tasks("dataonly") == set()
    assert fake_registry.supported_tasks("fake") == {TaskType.BINARY_CLASSIFICATION}


def test_component_requires_the_requested_kind(fake_registry: AdapterRegistry) -> None:
    with pytest.raises(ConfigError, match="provides no data adapter"):
        fake_registry.component("data", "fake", {})
    compute = fake_registry.component("compute", "fake", {})
    assert compute.name == "fake"


def test_loaded_plugins_reports_only_loaded_entries(fake_registry: AdapterRegistry) -> None:
    assert fake_registry.loaded_plugins() == {}
    plugin = fake_registry.get("fake")
    assert fake_registry.loaded_plugins() == {"fake": plugin}


def test_set_registry_swaps_the_process_registry(fake_registry: AdapterRegistry) -> None:
    from mbt.adapters import registry as registry_module

    original = registry_module._registry
    try:
        set_registry(fake_registry)
        assert get_registry() is fake_registry
    finally:
        set_registry(original)
    assert registry_module._registry is original


def test_missing_adapter_suggests_a_close_installed_name(fake_registry: AdapterRegistry) -> None:
    fake_registry.register(_plugin("nulladapter"))
    with pytest.raises(ConfigError, match="did you mean 'nulladapter'"):
        fake_registry.get("nulladaptr")
