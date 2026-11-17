"""Capabilities cover all five adapter kinds, and are declared rather than probed.

FEEDBACK v7 C-1: B-1 replaced core's ``hasattr`` probes for training adapters
and left eleven for the other four seams.
"""

from typing import Any

from mbt_adapter_base.capabilities import (
    _CAPABILITY_METHODS,
    AdapterKind,
    Capability,
    DeclaresCapabilities,
    capabilities_of,
    declaration_problems,
    kind_of,
    method_capabilities,
    supports,
)


def test_every_capability_belongs_to_exactly_one_kind() -> None:
    assert {kind_of(capability) for capability in Capability} <= set(AdapterKind)
    for capability in _CAPABILITY_METHODS:
        assert kind_of(capability) in AdapterKind


def test_each_non_training_kind_has_something_to_declare() -> None:
    for kind in (
        AdapterKind.TRACKING,
        AdapterKind.REGISTRY,
        AdapterKind.COMPUTE,
        AdapterKind.DATA,
    ):
        assert any(kind_of(capability) is kind for capability in Capability), kind


class _Tracker(DeclaresCapabilities):
    adapter_kind = AdapterKind.TRACKING

    def log_document(self, run: Any, path: Any) -> None: ...


def test_a_mixin_adapter_declares_exactly_the_methods_it_defines() -> None:
    assert capabilities_of(_Tracker()) == {Capability.LOG_DOCUMENT}
    assert supports(_Tracker(), Capability.LOG_DOCUMENT)
    assert not supports(_Tracker(), Capability.LOG_DIRECTORY)
    assert declaration_problems(_Tracker(), AdapterKind.TRACKING) == []


def test_a_kind_is_not_confused_by_a_method_of_another_kind() -> None:
    """A training adapter with a method called ``prepare`` has not declared a
    tracker's ``PREPARE``; ``kind`` is what keeps the two apart."""

    class _Trainer:
        def prepare(self) -> None: ...

    assert method_capabilities(_Trainer(), AdapterKind.TRAINING) == frozenset()
    assert method_capabilities(_Trainer()) == {Capability.PREPARE}


def test_a_multi_method_capability_needs_every_method() -> None:
    class _HalfScorer:
        def build_scoring_input(self, spec: Any, ctx: Any) -> Any: ...

    assert not supports(_HalfScorer(), Capability.BATCH_SCORING)

    class _Scorer(_HalfScorer):
        def open_predictions(self, output: Any) -> Any: ...

    assert supports(_Scorer(), Capability.BATCH_SCORING)


def test_an_adapter_that_predates_declared_capabilities_is_still_served() -> None:
    """The fallback is for a third-party adapter built against the pre-v5
    contract: the same rule the probes used, so it behaves as it did."""

    class _Old:
        def terminate(self, handle: Any, reason: str = "") -> None: ...

    assert supports(_Old(), Capability.TERMINATE)
    assert not supports(object(), Capability.TERMINATE)


def test_the_declaration_checker_finds_each_kind_of_disagreement() -> None:
    class _Overclaims(DeclaresCapabilities):
        adapter_kind = AdapterKind.TRACKING

        def capabilities(self, spec: Any = None) -> frozenset[Capability]:
            return frozenset({Capability.LOG_DOCUMENT})  # claims it, has no method

    assert any(
        "declares log_document but lacks" in p
        for p in declaration_problems(_Overclaims(), AdapterKind.TRACKING)
    )

    class _Underclaims:
        def capabilities(self, spec: Any = None) -> frozenset[Capability]:
            return frozenset()

        def set_version_tags(self, name: str, version: str, tags: dict[str, str]) -> None: ...

    assert any(
        "does not declare set_version_tags" in p
        for p in declaration_problems(_Underclaims(), AdapterKind.REGISTRY)
    )

    class _WrongKind:
        def capabilities(self, spec: Any = None) -> frozenset[Capability]:
            return frozenset({Capability.TERMINATE})

    assert any(
        "is a registry adapter but declares" in p
        for p in declaration_problems(_WrongKind(), AdapterKind.REGISTRY)
    )

    assert any(
        "declares no capabilities()" in p for p in declaration_problems(object(), AdapterKind.DATA)
    )
