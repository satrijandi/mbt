"""Training jobs export a monitoring baseline and register it (ADR-21)."""

import json
from pathlib import Path

from test_execution import MODEL, invoke

from mbt.adapters.registry import AdapterRegistry
from mbt_adapter_base.monitoring import read_baseline


def test_training_registers_baseline_and_hooks_hash(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    results = invoke(demo_project, fake_registry)
    assert results.exit_code() == 0
    model = next(r for r in results.results if r.unique_id == MODEL)
    assert model.registration is not None

    entry = json.loads((demo_project / "target/fake_registry/churn_model.json").read_text())[0]
    tags = entry["tags"]
    assert tags["mbt.hooks_hash"] == ""  # demo model has no hooks.py
    assert tags["mbt.baseline_uri"].startswith("file://")
    assert tags["mbt.baseline_format"] == "json"
    assert tags["mbt.baseline_content_hash"].startswith("sha256:")
    assert int(tags["mbt.baseline_size_bytes"]) > 0

    baseline = read_baseline(Path(tags["mbt.baseline_uri"].removeprefix("file://")))
    assert baseline.model_name == "churn_model"
    # post-hook features: everything except the target and the time column
    assert set(baseline.feature_columns) == {
        "user_id",
        "is_active",
        "tenure_days",
        "monthly_usage",
        "plan_type",
    }
    assert baseline.features["plan_type"].kind == "categorical"
    assert baseline.features["tenure_days"].kind == "numeric"
    assert baseline.score.n > 0
    assert len(baseline.score.quantiles) == 101
