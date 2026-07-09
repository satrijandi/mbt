"""Unit tests for the mbt-testing fake adapters themselves.

The fakes back most of the mbt-core suite indirectly; these tests pin the
scriptable behaviors that suite does not reach: failure injection, the
report ramp, slice filtering, registry misses, and every tuning dimension.
"""

from pathlib import Path
from typing import Any

import pytest
from mbt_testing.adapters import (
    FakeRegistryAdapter,
    FakeTrainingAdapter,
    FakeTuningEngine,
)

from mbt_adapter_base import (
    ArtifactRef,
    EvaluationProtocol,
    EvaluationSpec,
    MetricSpec,
    ModelSpec,
    ModelVersion,
    RunContext,
    Stage,
    TaskType,
    TuningSpec,
)
from mbt_adapter_base.compliance import tiny_binary_dataset


def _spec(**hyperparameters: Any) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": TaskType.BINARY_CLASSIFICATION,
            "adapter": "fake",
            "owner": "t@example.com",
            "dataset": "ref('d')",
            "target": "label",
            "hyperparameters": hyperparameters,
            "evaluation": EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
            "seed": 5,
        }
    )


def _ctx() -> RunContext:
    class _Null:
        def emit(self, event: object) -> None: ...

    return RunContext(
        run_id="t",
        unique_id="m",
        seed=5,
        target_name="dev",
        project_dir=".",
        vars={},
        events=_Null(),
    )


def _artifact() -> ArtifactRef:
    return ArtifactRef(uri="file:///x", format="fake_json", content_hash="sha256:0", size_bytes=1)


# -- FakeTrainingAdapter -----------------------------------------------------------


def test_fail_training_control_raises() -> None:
    adapter = FakeTrainingAdapter({})
    with pytest.raises(RuntimeError, match="fail_training=true"):
        adapter.train(_spec(fail_training=True), tiny_binary_dataset(), _ctx())


def test_train_with_report_ramps_to_the_final_value() -> None:
    adapter = FakeTrainingAdapter({})
    seen: list[tuple[int, float]] = []
    model = adapter.train_with_report(
        _spec(fake_metric_value=0.8, max_depth=2),
        tiny_binary_dataset(),
        _ctx(),
        lambda step, value: seen.append((step, value)),
    )
    assert [step for step, _ in seen] == list(range(10))
    values = [value for _, value in seen]
    assert values == sorted(values), "the report curve must ramp upward"
    assert values[-1] == pytest.approx(model.value)
    assert model.value == pytest.approx(0.8 + 2 * 1e-4)


def test_evaluate_skips_slice_columns_missing_from_the_table() -> None:
    adapter = FakeTrainingAdapter({})
    data = tiny_binary_dataset()
    model = adapter.train(_spec(fake_metric_value=0.6), data, _ctx())
    metrics = [MetricSpec(name="roc_auc", kind="builtin")]
    results = adapter.evaluate(model, data, "test", metrics, slices=["not_a_column", "f_binary"])
    assert results.metrics == {"roc_auc": pytest.approx(model.value)}
    assert results.slices  # the real column produced groups
    assert all(key.startswith("f_binary=") for key in results.slices)


def test_fake_adapter_reports_no_nondeterminism() -> None:
    assert FakeTrainingAdapter({}).nondeterminism_warnings(_spec()) == []


# -- FakeRegistryAdapter -----------------------------------------------------------


def test_get_version_misses_return_none(tmp_path: Path) -> None:
    registry = FakeRegistryAdapter({"root": str(tmp_path / "registry")})
    assert registry.get_version("never_registered", "1") is None
    registry.register(_artifact(), "m", {"k": "v"})
    assert registry.get_version("m", "2") is None
    found = registry.get_version("m", "1")
    assert found is not None and found.tags == {"k": "v"}


def test_transition_of_unknown_version_is_a_lookup_error(tmp_path: Path) -> None:
    registry = FakeRegistryAdapter({"root": str(tmp_path / "registry")})
    registry.register(_artifact(), "m", {})
    ghost = ModelVersion(name="m", version="99", artifact=_artifact())
    with pytest.raises(LookupError, match="version 99 of 'm' not found"):
        registry.transition(ghost, Stage.STAGING)


# -- FakeTuningEngine ----------------------------------------------------------------


def test_fake_tuning_samples_every_dimension_type() -> None:
    spec = TuningSpec.model_validate(
        {
            "engine": "fake",
            "n_trials": 6,
            "search_space": {
                "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
                "max_depth": {"type": "int", "low": 2, "high": 6},
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                "subsample": {"type": "uniform", "low": 0.5, "high": 1.0},
            },
            "objective": {"metric": "pr_auc", "direction": "maximize"},
        }
    )
    proposals: list[dict[str, Any]] = []

    def objective(params: dict[str, Any]) -> float:
        proposals.append(params)
        return params["subsample"]

    result = FakeTuningEngine({}).tune(spec, objective, n_trials=6, seed=7)
    assert result.n_trials == 6
    assert result.best_value == pytest.approx(max(p["subsample"] for p in proposals))
    for params in proposals:
        assert params["booster"] in ("gbtree", "dart")
        assert isinstance(params["max_depth"], int) and 2 <= params["max_depth"] <= 6
        assert 1e-4 <= params["learning_rate"] <= 1e-1
        assert 0.5 <= params["subsample"] <= 1.0


def test_fake_tuning_minimize_direction_tracks_the_lowest_value() -> None:
    spec = TuningSpec.model_validate(
        {
            "engine": "fake",
            "n_trials": 4,
            "search_space": {"quality": {"type": "uniform", "low": 0.0, "high": 1.0}},
            "objective": {"metric": "logloss", "direction": "minimize"},
        }
    )
    seen: list[float] = []

    def objective(params: dict[str, Any]) -> float:
        seen.append(params["quality"])
        return params["quality"]

    result = FakeTuningEngine({}).tune(spec, objective, n_trials=4, seed=3)
    assert result.best_value == pytest.approx(min(seen))
