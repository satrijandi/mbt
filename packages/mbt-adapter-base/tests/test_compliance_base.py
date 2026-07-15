"""Compliance-suite base-class plumbing, exercised with fakes (no framework)."""

from pathlib import Path
from typing import Any

import pytest

from mbt_adapter_base.compliance.suite import (
    PredictionStoreCompliance,
    TrainingAdapterCompliance,
    _NullSink,
)
from mbt_adapter_base.interchange import DatasetProfile
from mbt_adapter_base.specs import ModelSpec


class _ResolveOnlyAdapter:
    """Just enough adapter surface for the resolve_auto compliance check."""

    name = "fake_resolve_only"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def resolve_auto(self, spec: ModelSpec, profile: DatasetProfile) -> ModelSpec:
        return spec


class _NoAutoCompliance(TrainingAdapterCompliance):
    """A suite subclass whose adapter declares no AUTO hyperparameter."""

    adapter_factory = _ResolveOnlyAdapter
    plugin_module = "mbt_adapter_base"
    framework_modules = ()


def test_null_sink_swallows_events() -> None:
    assert _NullSink().emit({"event": "job_started"}) is None


def test_resolve_auto_check_without_auto_hyperparameter() -> None:
    _NoAutoCompliance().test_resolve_auto_idempotent_and_no_sentinels()


def test_prediction_store_compliance_requires_make_store(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        PredictionStoreCompliance().make_store(tmp_path)


class _NoImportanceAdapter:
    """An adapter without the optional feature_importance capability."""

    name = "fake_no_importance"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config


class _NoImportanceCompliance(TrainingAdapterCompliance):
    adapter_factory = _NoImportanceAdapter
    plugin_module = "mbt_adapter_base"
    framework_modules = ()


def test_feature_importance_check_skips_without_the_method() -> None:
    with pytest.raises(pytest.skip.Exception, match="optional"):
        _NoImportanceCompliance().test_feature_importance_is_normalized_when_supported()


class _EmptyImportanceAdapter:
    """An adapter whose winning model cannot attribute importance ({})."""

    name = "fake_empty_importance"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def train(self, spec: Any, data: Any, ctx: Any) -> object:
        return object()

    def feature_importance(self, model: Any) -> dict[str, float]:
        return {}


class _EmptyImportanceCompliance(TrainingAdapterCompliance):
    adapter_factory = _EmptyImportanceAdapter
    plugin_module = "mbt_adapter_base"
    framework_modules = ()


def test_feature_importance_check_accepts_the_empty_escape_hatch() -> None:
    # {} is the documented outcome for unattributable models (ensemble leaders).
    _EmptyImportanceCompliance().test_feature_importance_is_normalized_when_supported()
