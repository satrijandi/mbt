"""Cross-adapter categorical parity (F25's structural remainder).

The compliance suite proves each adapter LEARNS from a categorical feature in
isolation; these tests compare adapters AGAINST EACH OTHER on the same
fixtures, so a categorical-handling regression in one adapter (or a new
adapter shipping ordinal handling while claiming native parity, F24) fails
loudly instead of passing "correct by construction".

Two fixtures probe two different failure modes:

- ``tiny_mixed_dataset`` (3 levels): ANY sane encoding can represent 3 levels,
  so all four adapters must land in one tight roc_auc band - this catches a
  broken categorical path.
- a high-cardinality alternating-rate fixture (26 levels whose positive rates
  alternate by frequency rank): native unordered-categorical splits isolate
  the levels; frequency-ranked ORDINAL codes need ~one threshold per level,
  which a small GBT cannot afford. This is exactly the F24 representation
  divergence, so the native trio (xgboost, lightgbm, h2o) is held to a tight
  band while spark is held only to a documented learning floor - if spark
  gains a native-categorical path one day, the floor still passes and the gap
  simply closes.
"""

import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from mbt_adapter_base import MetricSpec, ModelSpec, RunContext
from mbt_adapter_base.compliance.suite import tiny_mixed_dataset
from mbt_adapter_base.datasets import InMemoryDatasetHandle

ROC_AUC = [MetricSpec(name="roc_auc", kind="builtin", greater_is_better=True)]

#: Native-trio band on either fixture: measured spreads are ~0.02 (mixed) and
#: ~0.03 (high-cardinality); 0.1 leaves honest headroom without letting a
#: broken encoding (which drops to ~coin-flip) slip through.
NATIVE_BAND = 0.1

#: Spark's documented F24 floor on the high-cardinality fixture: ordinal codes
#: still LEARN (well above the 0.5 coin flip), but are not held to the native
#: band because threshold splits on arbitrary codes are a different model
#: family (see mbt-spark README "Categorical parity caveat").
SPARK_HIGH_CARDINALITY_FLOOR = 0.55


def high_cardinality_dataset(
    n_rows: int = 1500, levels: int = 26, seed: int = 5
) -> InMemoryDatasetHandle:
    """A categorical-signal dataset ordinal encodings cannot cheaply represent:
    level frequencies decay (so frequency-ranked codes track the level index)
    while positive rates ALTERNATE by index - separating them needs unordered
    subset splits, not a few thresholds."""
    rng = np.random.default_rng(seed)
    weights = np.array([1.0 / (i + 1) for i in range(levels)])
    weights /= weights.sum()
    idx = rng.choice(levels, size=n_rows, p=weights)
    rates = np.where(idx % 2 == 1, 0.9, 0.1)
    table = pa.table(
        {
            "segment": [f"seg_{i:02d}" for i in idx],
            "noise": rng.normal(size=n_rows),
            "label": (rng.random(n_rows) < rates).astype("int64"),
        }
    )
    cut = int(n_rows * 0.75)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, cut), "test": table.slice(cut)},
        snapshot_id="sha256:parity-high-cardinality",
        label_column="label",
    )


def _spec(adapter: str, hyperparameters: dict[str, Any]) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": f"parity_{adapter}",
            "task": "binary_classification",
            "adapter": adapter,
            "owner": "parity@example.com",
            "dataset": "ref('parity')",
            "target": "label",
            "hyperparameters": hyperparameters,
            "evaluation": {"protocol": {"split": "random"}, "metrics": ["roc_auc"]},
            "seed": 7,
        }
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="parity",
        unique_id="model.parity.parity",
        seed=7,
        target_name="parity",
        project_dir=".",
        vars={},
        events=None,
    )


def _auc(adapter: Any, spec: ModelSpec, data: InMemoryDatasetHandle) -> float:
    model = adapter.train(spec, data, _ctx())
    return float(adapter.evaluate(model, data, "test", ROC_AUC).metrics["roc_auc"])


def _tree_aucs(data: InMemoryDatasetHandle) -> dict[str, float]:
    from mbt_lightgbm.adapter import LightGBMTrainingAdapter
    from mbt_xgboost.adapter import XGBoostTrainingAdapter

    return {
        "xgboost": _auc(
            XGBoostTrainingAdapter({}),
            _spec("xgboost", {"max_depth": 3, "n_estimators": 30, "learning_rate": 0.2}),
            data,
        ),
        "lightgbm": _auc(
            LightGBMTrainingAdapter({}),
            _spec("lightgbm", {"num_leaves": 15, "n_estimators": 30, "learning_rate": 0.2}),
            data,
        ),
    }


def test_native_tree_adapters_agree_on_categorical_signal() -> None:
    """xgboost and lightgbm (both native-categorical via the shared encoding)
    must land in one band on both fixtures - fast-tier half of the F25
    parity assertion."""
    for data in (tiny_mixed_dataset(), high_cardinality_dataset()):
        aucs = _tree_aucs(data)
        assert all(value > 0.8 for value in aucs.values()), aucs
        assert max(aucs.values()) - min(aucs.values()) <= NATIVE_BAND, aucs


@pytest.mark.e2e
@pytest.mark.skipif(shutil.which("java") is None, reason="H2O/Spark need a JVM")
def test_jvm_adapters_parity_and_the_documented_spark_gap() -> None:
    """e2e half: h2o joins the native band on both fixtures; spark joins it on
    the low-cardinality fixture (3 levels are representable by thresholds) but
    is held only to the documented F24 learning floor on the high-cardinality
    one."""
    from mbt_h2o.adapter import H2OAutoMLAdapter
    from mbt_spark.training import SparkMLTrainingAdapter

    h2o_spec = _spec("h2o", {"max_models": 2, "include_algos": ["GLM"], "nfolds": 0})
    spark_spec = _spec("spark", {"max_iter": 5, "max_depth": 3})
    h2o_adapter = H2OAutoMLAdapter({})
    spark_adapter = SparkMLTrainingAdapter({})

    mixed = tiny_mixed_dataset()
    band = {**_tree_aucs(mixed)}
    band["h2o"] = _auc(h2o_adapter, h2o_spec, mixed)
    band["spark"] = _auc(spark_adapter, spark_spec, mixed)
    assert all(value > 0.8 for value in band.values()), band
    assert max(band.values()) - min(band.values()) <= NATIVE_BAND, band

    high = high_cardinality_dataset()
    natives = {**_tree_aucs(high)}
    natives["h2o"] = _auc(h2o_adapter, h2o_spec, high)
    assert all(value > 0.8 for value in natives.values()), natives
    assert max(natives.values()) - min(natives.values()) <= NATIVE_BAND, natives
    spark_auc = _auc(spark_adapter, spark_spec, high)
    assert spark_auc >= SPARK_HIGH_CARDINALITY_FLOOR, spark_auc
