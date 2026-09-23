"""What an adapter can optionally do, declared rather than probed (B-1).

Four capability protocols were declared in ``protocols.py`` - and no dispatch
went through them. What decided was 16 ``hasattr`` probes across ``execute/``
plus two ``getattr`` probes in ``quality/checks.py``. Deleting all four protocol
declarations changed nothing at runtime: they were documentation typed as code,
mypy never checked the dispatch, and a renamed method silently disabled a
capability.

Three more flags - ``supports_calibration``, ``supports_monotonic_constraints``,
``supports_categorical_pooling`` - were on no protocol at all and were probed by
the parser by name. The compliance suite gated a test on
``getattr(adapter, "supports_calibration", False)``, so a typo in the class
variable silently SKIPPED the test instead of failing it.

So capability is one method now: ``capabilities(spec) -> frozenset[Capability]``.

It takes the spec because capability is not always a property of the adapter.
sklearn advertised a blunt ``supports_monotonic_constraints = True`` and then
raised in ``validate`` for the estimators that cannot honour one - the flag had
no granularity, so it could not say "yes, for the histogram booster". With the
spec in hand it can.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mbt_adapter_base.specs import ModelSpec


class Capability(StrEnum):
    """One optional thing a training adapter may be able to do."""

    #: Per-round tuning progress, for pruning: ``train_with_report``.
    TRAIN_WITH_REPORT = "train_with_report"
    #: Model-intrinsic per-feature importance: ``feature_importance``.
    FEATURE_IMPORTANCE = "feature_importance"
    #: SHAP-based global importance, data-grounded: ``shap_importance``.
    SHAP_IMPORTANCE = "shap_importance"
    #: Per-row local attribution: ``explain``. REQUIRED when a scoring node
    #: sets ``output.explain_top_k``.
    EXPLAIN = "explain"
    #: Post-hoc probability calibration (``calibration:`` on the spec).
    CALIBRATION = "calibration"
    #: Monotone constraints on named features (ADR-27).
    MONOTONIC_CONSTRAINTS = "monotonic_constraints"
    #: Pooling rare categorical levels by ``min_frequency`` (ADR-27).
    CATEGORICAL_POOLING = "categorical_pooling"
    #: Reporting how many rounds a fit actually used: ``best_iteration``.
    #: What lets tuning carry the complexity its trials CHOSE into the final
    #: fit, when that fit has no validation split to stop on (D-2).
    BEST_ITERATION = "best_iteration"


#: Capability -> the method that used to be ``hasattr``-probed for it. Used
#: only by the compatibility fallback below.
_CAPABILITY_METHODS = {
    Capability.TRAIN_WITH_REPORT: "train_with_report",
    Capability.FEATURE_IMPORTANCE: "feature_importance",
    Capability.SHAP_IMPORTANCE: "shap_importance",
    Capability.EXPLAIN: "explain",
    Capability.BEST_ITERATION: "best_iteration",
}

#: Capability -> the class variable that used to declare it.
_CAPABILITY_FLAGS = {
    Capability.CALIBRATION: "supports_calibration",
    Capability.MONOTONIC_CONSTRAINTS: "supports_monotonic_constraints",
    Capability.CATEGORICAL_POOLING: "supports_categorical_pooling",
}


def method_capabilities(adapter: Any) -> frozenset[Capability]:
    """The capabilities implied by which optional methods ``adapter`` defines.

    This is the sensible default for an adapter whose capabilities really are a
    property of the class - which is every in-repo adapter but sklearn. It is
    the implementation ``BaseTrainingAdapter.capabilities`` uses, not a probe
    core performs: the difference is that an adapter can override it, and that
    the answer is computed once in a place a reader can find.
    """
    return frozenset(
        capability
        for capability, method in _CAPABILITY_METHODS.items()
        if callable(getattr(adapter, method, None))
    )


def capabilities_of(adapter: Any, spec: "ModelSpec | None" = None) -> frozenset[Capability]:
    """What ``adapter`` can do, for the ``spec`` when one is given.

    Core calls THIS rather than probing, so a renamed method is a capability
    the adapter stops declaring rather than one that silently disappears.

    The fallback is for a third-party adapter built against the pre-v5 contract,
    which declared capabilities by method presence and ``supports_*`` class
    variables. It is deliberately the same rule those probes used, so such an
    adapter behaves exactly as it did before.
    """
    declared = getattr(adapter, "capabilities", None)
    if callable(declared):
        return frozenset(declared(spec))
    legacy = set(method_capabilities(adapter))
    legacy |= {
        capability
        for capability, flag in _CAPABILITY_FLAGS.items()
        if getattr(adapter, flag, False)
    }
    return frozenset(legacy)


def supports(adapter: Any, capability: Capability, spec: "ModelSpec | None" = None) -> bool:
    """Whether ``adapter`` can do ``capability`` (for ``spec``, when given)."""
    return capability in capabilities_of(adapter, spec)


__all__ = [
    "Capability",
    "capabilities_of",
    "method_capabilities",
    "supports",
]
