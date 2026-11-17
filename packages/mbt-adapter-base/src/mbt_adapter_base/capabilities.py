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

B-1 did that for ONE of the five adapter kinds. The tracking, registry,
compute and data adapters kept their ``hasattr`` probes in ``execute/`` and
``quality/``, so a renamed ``log_document`` silently stopped logging the
inference config, a missing ``terminate`` made ``--fail-fast`` leave jobs
running, and a data adapter without ``open_predictions`` was refused by a probe
that read nothing but method names (FEEDBACK v7 C-1). The same enum now names
what every kind can optionally do, ``AdapterKind`` says which kind a capability
belongs to, and ``test_no_hasattr_probe_guard.py`` keeps core from probing again.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from mbt_adapter_base.specs import ModelSpec


class AdapterKind(StrEnum):
    """The five seams an adapter plugs into; a capability belongs to exactly one."""

    TRAINING = "training"
    TRACKING = "tracking"
    REGISTRY = "registry"
    COMPUTE = "compute"
    DATA = "data"


class Capability(StrEnum):
    """One optional thing an adapter may be able to do.

    Each member belongs to one ``AdapterKind`` (``kind_of``); a tracker declares
    the tracking ones and says nothing of the rest.
    """

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

    # -- tracking adapters -------------------------------------------------------
    #: ``prepare()``: warm the backend (e.g. MLflow's sqlite migrations) before
    #: parallel jobs hit it at once.
    PREPARE = "prepare"
    #: ``log_trial(run, index, params, value)``: tuning history as nested runs.
    LOG_TRIAL = "log_trial"
    #: ``log_document(run, path)``: upload one local file mbt wrote, such as
    #: the inference config.
    LOG_DOCUMENT = "log_document"
    #: ``log_directory(run, local_dir, artifact_path)``: upload a directory mbt
    #: wrote (the training report, the run log) keeping its layout (ADR-30).
    LOG_DIRECTORY = "log_directory"

    # -- registry adapters -------------------------------------------------------
    #: ``set_version_tags(name, version, tags)``: record facts learned after
    #: registration, such as the pre-deploy check's verdict (ADR-30).
    SET_VERSION_TAGS = "set_version_tags"
    #: ``list_versions(name)``: every registered version of a model, newest
    #: first. What ``mbt registry ls`` reads; promotion and rollback address
    #: versions one at a time and never needed it.
    LIST_VERSIONS = "list_versions"

    # -- compute adapters --------------------------------------------------------
    #: ``terminate(handle, reason)``: the kill seam ``--fail-fast`` and Ctrl-C use.
    TERMINATE = "terminate"

    # -- data adapters -----------------------------------------------------------
    #: ``build_scoring_input`` + ``open_predictions``: batch scoring, contract 1.1
    #: (ADR-20/21). An adapter built against 1.0 does not declare it.
    BATCH_SCORING = "batch_scoring"
    #: ``count_source_duplicates`` + ``read_source_distinct``: the source-level
    #: checks (``unique`` with ``source:``, ``relationships``).
    SOURCE_CHECKS = "source_checks"


_KINDS: dict[Capability, AdapterKind] = {
    Capability.TRAIN_WITH_REPORT: AdapterKind.TRAINING,
    Capability.FEATURE_IMPORTANCE: AdapterKind.TRAINING,
    Capability.SHAP_IMPORTANCE: AdapterKind.TRAINING,
    Capability.EXPLAIN: AdapterKind.TRAINING,
    Capability.CALIBRATION: AdapterKind.TRAINING,
    Capability.MONOTONIC_CONSTRAINTS: AdapterKind.TRAINING,
    Capability.CATEGORICAL_POOLING: AdapterKind.TRAINING,
    Capability.BEST_ITERATION: AdapterKind.TRAINING,
    Capability.PREPARE: AdapterKind.TRACKING,
    Capability.LOG_TRIAL: AdapterKind.TRACKING,
    Capability.LOG_DOCUMENT: AdapterKind.TRACKING,
    Capability.LOG_DIRECTORY: AdapterKind.TRACKING,
    Capability.SET_VERSION_TAGS: AdapterKind.REGISTRY,
    Capability.LIST_VERSIONS: AdapterKind.REGISTRY,
    Capability.TERMINATE: AdapterKind.COMPUTE,
    Capability.BATCH_SCORING: AdapterKind.DATA,
    Capability.SOURCE_CHECKS: AdapterKind.DATA,
}


def kind_of(capability: Capability) -> AdapterKind:
    """The adapter kind ``capability`` belongs to."""
    return _KINDS[capability]


#: Capability -> the methods that used to be ``hasattr``-probed for it, ALL of
#: which must be callable. Used by the compatibility fallback below and by
#: ``declaration_problems``; a capability the class variables declare
#: (``_CAPABILITY_FLAGS``) is not in it.
_CAPABILITY_METHODS: dict[Capability, tuple[str, ...]] = {
    Capability.TRAIN_WITH_REPORT: ("train_with_report",),
    Capability.FEATURE_IMPORTANCE: ("feature_importance",),
    Capability.SHAP_IMPORTANCE: ("shap_importance",),
    Capability.EXPLAIN: ("explain",),
    Capability.BEST_ITERATION: ("best_iteration",),
    Capability.PREPARE: ("prepare",),
    Capability.LOG_TRIAL: ("log_trial",),
    Capability.LOG_DOCUMENT: ("log_document",),
    Capability.LOG_DIRECTORY: ("log_directory",),
    Capability.SET_VERSION_TAGS: ("set_version_tags",),
    Capability.LIST_VERSIONS: ("list_versions",),
    Capability.TERMINATE: ("terminate",),
    Capability.BATCH_SCORING: ("build_scoring_input", "open_predictions"),
    Capability.SOURCE_CHECKS: ("count_source_duplicates", "read_source_distinct"),
}

#: Capability -> the class variable that used to declare it.
_CAPABILITY_FLAGS = {
    Capability.CALIBRATION: "supports_calibration",
    Capability.MONOTONIC_CONSTRAINTS: "supports_monotonic_constraints",
    Capability.CATEGORICAL_POOLING: "supports_categorical_pooling",
}


def _implements(adapter: Any, methods: tuple[str, ...]) -> bool:
    return all(callable(getattr(adapter, method, None)) for method in methods)


def method_capabilities(adapter: Any, kind: AdapterKind | None = None) -> frozenset[Capability]:
    """The capabilities implied by which optional methods ``adapter`` defines.

    ``kind`` limits the answer to one seam's capabilities, which is what an
    adapter of that kind should pass: a training adapter that happens to define
    a method called ``prepare`` has not declared ``Capability.PREPARE``.

    This is the sensible default for an adapter whose capabilities really are a
    property of the class - which is every in-repo adapter but sklearn. It is
    the implementation ``DeclaresCapabilities`` and
    ``ArrowTrainingAdapter.capabilities`` use, not a probe core performs: the
    difference is that an adapter can override it, and that the answer is
    computed once in a place a reader can find.
    """
    return frozenset(
        capability
        for capability, methods in _CAPABILITY_METHODS.items()
        if (kind is None or _KINDS[capability] is kind) and _implements(adapter, methods)
    )


class DeclaresCapabilities:
    """Mixin for a non-training adapter: declare what its methods implement.

    Set ``adapter_kind`` on the class. The answer is "the optional methods this
    class defines, for this kind", so adding ``log_document`` to a tracker
    declares it, and the compliance suite fails if the two ever disagree.
    """

    adapter_kind: ClassVar[AdapterKind]

    def capabilities(self, spec: "ModelSpec | None" = None) -> frozenset[Capability]:
        return method_capabilities(self, self.adapter_kind)


def declaration_problems(adapter: Any, kind: AdapterKind) -> list[str]:
    """Why ``adapter``'s capability declaration is wrong, or ``[]`` when it is right.

    The compliance suites call this. Both directions, as B-1 asked of training
    adapters: a declared capability needs its methods, and a method of a
    capability must be declared, or core will never call it. An adapter with no
    ``capabilities`` at all is a problem too: core still serves it through the
    fallback below, but only so a pre-v5 third-party adapter keeps working.
    """
    name = type(adapter).__name__
    if not callable(getattr(adapter, "capabilities", None)):
        return [f"{name} declares no capabilities(); implement it (DeclaresCapabilities)"]
    declared = capabilities_of(adapter)
    problems = []
    for capability, methods in _CAPABILITY_METHODS.items():
        if _KINDS[capability] is not kind:
            continue
        present = _implements(adapter, methods)
        if capability in declared and not present:
            problems.append(f"{name} declares {capability} but lacks callable {list(methods)}")
        elif present and capability not in declared:
            problems.append(
                f"{name} defines {list(methods)} but does not declare {capability}, "
                "so core will never call it"
            )
    foreign = sorted(str(capability) for capability in declared if _KINDS[capability] is not kind)
    if foreign:
        problems.append(f"{name} is a {kind} adapter but declares {foreign}")
    return problems


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
    "AdapterKind",
    "Capability",
    "DeclaresCapabilities",
    "capabilities_of",
    "declaration_problems",
    "kind_of",
    "method_capabilities",
    "supports",
]
