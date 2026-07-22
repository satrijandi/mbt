"""The optional-capability protocols (F27).

These are ``@runtime_checkable`` so a test or the compliance suite can assert an
adapter *advertises* a capability - the same presence core probes with
``hasattr``. runtime_checkable only checks the method NAME; the signature teeth
are static, enforced by the ``_capability_conformance`` mypy variables in each
adapter that implements a capability (see mbt-xgboost / mbt-lightgbm).
"""

from typing import Any

from mbt_adapter_base import (
    SupportsExplain,
    SupportsFeatureImportance,
    SupportsShapImportance,
    SupportsTrainWithReport,
)


class _FullyCapable:
    """A stub advertising every optional capability (bodies irrelevant here)."""

    def feature_importance(self, model: Any) -> dict[str, float]:
        return {}

    def shap_importance(self, model: Any, data: Any, split: str) -> dict[str, float]:
        return {}

    def explain(self, model: Any, data: Any, split: str, top_k: int) -> list[str]:
        return []

    def train_with_report(self, spec: Any, data: Any, ctx: Any, report: Any) -> Any:
        return object()


class _PlainAdapter:
    """A stub advertising none of the optional capabilities."""


def test_capability_protocols_detect_presence_by_the_probed_method_name() -> None:
    capable = _FullyCapable()
    assert isinstance(capable, SupportsFeatureImportance)
    assert isinstance(capable, SupportsShapImportance)
    assert isinstance(capable, SupportsExplain)
    assert isinstance(capable, SupportsTrainWithReport)


def test_capability_protocols_reject_an_adapter_missing_the_method() -> None:
    plain = _PlainAdapter()
    assert not isinstance(plain, SupportsFeatureImportance)
    assert not isinstance(plain, SupportsShapImportance)
    assert not isinstance(plain, SupportsExplain)
    assert not isinstance(plain, SupportsTrainWithReport)
