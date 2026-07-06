"""Tuning semantics with the fake engine (S8-03, ADR-8, FR-TUNE-01..04)."""

from pathlib import Path

from conftest import TEST_ANCHOR
from test_execution import MODEL, invoke

from mbt.adapters.registry import AdapterRegistry

TUNING_BLOCK = """
    tuning:
      engine: fake
      n_trials: 10
      search_space:
        fake_metric_value: {type: uniform, low: 0.45, high: 0.95}
        max_depth: {type: int, low: 2, high: 8}
      objective: {metric: pr_auc, direction: maximize}
"""


def _add_tuning(demo_project: Path) -> None:
    model_yml = demo_project / "models/churn_model.yml"
    text = model_yml.read_text().replace(
        "    evaluation:", TUNING_BLOCK.rstrip() + "\n    evaluation:"
    )
    model_yml.write_text(text)


def test_tuning_respects_target_cap_and_is_seeded(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    _add_tuning(demo_project)
    first = invoke(
        demo_project, fake_registry, cli_vars={"max_tuning_trials": 3}
    )
    assert first.exit_code() == 0
    model_first = {r.unique_id: r for r in first.results}[MODEL]

    second = invoke(
        demo_project, fake_registry, cli_vars={"max_tuning_trials": 3}
    )
    model_second = {r.unique_id: r for r in second.results}[MODEL]
    # same seed -> same proposals -> identical best params and metrics
    assert model_first.metrics == model_second.metrics


def test_tuning_uncapped_uses_declared_trials(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch
) -> None:
    _add_tuning(demo_project)
    calls = {"count": 0}

    from mbt_testing.adapters import FakeTrainingAdapter

    original_train = FakeTrainingAdapter.train

    def counting_train(self, spec, data, ctx):
        calls["count"] += 1
        return original_train(self, spec, data, ctx)

    monkeypatch.setattr(FakeTrainingAdapter, "train", counting_train)
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 0
    # 10 trials + 1 final fit (inline compute keeps this observable)
    assert calls["count"] == 11


def test_tuning_never_reads_the_test_split(
    demo_project: Path, fake_registry: AdapterRegistry, monkeypatch
) -> None:
    """ADR-8: trials train on train and evaluate on validation only."""
    _add_tuning(demo_project)
    splits_evaluated: list[str] = []

    from mbt_testing.adapters import FakeTrainingAdapter

    original_evaluate = FakeTrainingAdapter.evaluate

    def spying_evaluate(self, model, data, split, metrics, slices=None):
        splits_evaluated.append(split)
        return original_evaluate(self, model, data, split, metrics, slices)

    monkeypatch.setattr(FakeTrainingAdapter, "evaluate", spying_evaluate)
    results = invoke(demo_project, fake_registry, cli_vars={"max_tuning_trials": 4})
    assert results.exit_code() == 0
    assert splits_evaluated.count("validation") == 4  # one per trial
    assert splits_evaluated.count("test") == 1  # final evaluation only
    assert splits_evaluated.index("test") == len(splits_evaluated) - 1
