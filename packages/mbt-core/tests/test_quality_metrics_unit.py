"""Unit tests for metric name resolution (mbt/quality/metrics.py)."""

from mbt.config.tasks import get_task_schema
from mbt.contracts import MetricSpec, TaskType
from mbt.quality.metrics import metric_direction, resolve_metric

SCHEMA = get_task_schema(TaskType.BINARY_CLASSIFICATION)


def test_declared_builtin_with_plain_name_resolves() -> None:
    declared = {"my_auc": MetricSpec(name="pr_auc", kind="builtin")}
    outcome = resolve_metric("my_auc", declared, SCHEMA, has_hooks=False)
    assert isinstance(outcome, MetricSpec) and outcome.name == "pr_auc"


def test_declared_builtin_with_sugar_name_resolves() -> None:
    declared = {"strict_recall": MetricSpec(name="recall_at_precision_0.9", kind="builtin")}
    outcome = resolve_metric("strict_recall", declared, SCHEMA, has_hooks=False)
    assert isinstance(outcome, MetricSpec) and outcome.name == "recall_at_precision_0.9"


def test_declared_builtin_with_bogus_name_is_an_error() -> None:
    declared = {"bad": MetricSpec(name="not_a_metric", kind="builtin")}
    outcome = resolve_metric("bad", declared, SCHEMA, has_hooks=False)
    assert isinstance(outcome, str)
    assert "declared builtin in metrics.yml" in outcome


def test_declared_hook_metric_passes_through() -> None:
    declared = {"custom": MetricSpec(name="custom", kind="hook")}
    outcome = resolve_metric("custom", declared, SCHEMA, has_hooks=False)
    assert isinstance(outcome, MetricSpec) and outcome.kind == "hook"


def test_unknown_name_with_hooks_becomes_hook_metric() -> None:
    outcome = resolve_metric("bespoke_metric", {}, SCHEMA, has_hooks=True)
    assert isinstance(outcome, MetricSpec)
    assert outcome.kind == "hook" and outcome.greater_is_better


def test_unknown_name_without_hooks_is_an_error() -> None:
    outcome = resolve_metric("bespoke_metric", {}, SCHEMA, has_hooks=False)
    assert isinstance(outcome, str) and "unknown metric" in outcome


def test_metric_direction_uses_resolved_specs_first() -> None:
    specs = [MetricSpec(name="my_loss", kind="hook", greater_is_better=False)]
    assert metric_direction("my_loss", specs) is False


def test_metric_direction_falls_back_to_sugar_and_defaults() -> None:
    assert metric_direction("recall_at_precision_0.9", []) is True
    assert metric_direction("logloss", []) is False
    assert metric_direction("pr_auc", []) is True


def test_unknown_metric_suggests_a_close_candidate() -> None:
    outcome = resolve_metric("pr_au", {}, SCHEMA, has_hooks=False)
    assert isinstance(outcome, str)
    assert "did you mean 'pr_auc'" in outcome
    assert "unknown metric" in outcome


def test_regression_metric_direction_and_resolution() -> None:
    from mbt.config.tasks import get_task_schema
    from mbt.quality.metrics import default_direction

    assert default_direction("rmse") is False
    assert default_direction("mae") is False
    assert default_direction("mape") is False
    assert default_direction("r2") is True

    reg = get_task_schema(TaskType.REGRESSION)
    rmse = resolve_metric("rmse", {}, reg, has_hooks=False)
    assert isinstance(rmse, MetricSpec) and rmse.greater_is_better is False
    r2 = resolve_metric("r2", {}, reg, has_hooks=False)
    assert isinstance(r2, MetricSpec) and r2.greater_is_better is True
    # a binary metric is not a builtin for regression
    bad = resolve_metric("roc_auc", {}, reg, has_hooks=False)
    assert isinstance(bad, str) and "unknown metric" in bad
