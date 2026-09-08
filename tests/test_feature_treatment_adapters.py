"""Cross-adapter behaviour of declarative feature treatment (ADR-27).

The unit tests prove the transform math; these prove the two halves that only
exist once a real framework is involved:

- a monotone constraint actually bends the fitted model, on every adapter that
  claims to support one;
- a DS-declared int-coded categorical reaches each adapter as a category,
  which is the whole reason the declaration exists (an ordinal code trained as
  a magnitude cannot represent a non-monotone level effect).

Both run through the same core seam a real job uses - `TransformedDatasetHandle`
applies the treatment, exactly as `mbt run` does - rather than hand-feeding
pre-treated tables, so a wiring regression in core fails here too.
"""

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from mbt.contracts import HookContext, ModelSpec, RunContext
from mbt.events import get_bus
from mbt.execute.handles import TransformedDatasetHandle
from mbt_adapter_base import MetricSpec
from mbt_adapter_base.datasets import InMemoryDatasetHandle

ROC_AUC = [MetricSpec(name="roc_auc", kind="builtin", greater_is_better=True)]

#: A prediction sweep is monotone up to float noise, not exactly.
MONOTONE_TOLERANCE = 1e-9


def _spec(adapter: str, features: dict[str, Any], **hyperparameters: Any) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": f"treated_{adapter}",
            "task": "binary_classification",
            "adapter": adapter,
            "owner": "ds@example.com",
            "dataset": "ref('treated')",
            "target": "label",
            "features": features,
            "hyperparameters": hyperparameters,
            "evaluation": {"protocol": {"split": "random"}, "metrics": ["roc_auc"]},
            "seed": 7,
        }
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="treated",
        unique_id="model.treated.treated",
        seed=7,
        target_name="treated",
        project_dir=".",
        vars={},
        events=None,
    )


def _treated(base: InMemoryDatasetHandle, spec: ModelSpec) -> TransformedDatasetHandle:
    """The handle a training job builds, so treatment is applied by the code
    under test rather than by the test."""

    def hook_ctx(split: str) -> HookContext:
        return HookContext(spec=spec, profile=None, split=split, logger=get_bus())

    return TransformedDatasetHandle(base, spec, None, hook_ctx, None)


# -- monotone constraints --------------------------------------------------------


def _u_shaped_dataset(n_rows: int = 3000, seed: int = 11) -> InMemoryDatasetHandle:
    """A deliberately NON-monotone relationship: the positive rate is high at
    both ends of ``x`` and low in the middle. An unconstrained booster fits the
    dip; a monotone-increasing one may not."""
    rng = np.random.default_rng(seed)
    x = rng.random(n_rows)
    rates = np.where((x < 0.3) | (x >= 0.7), 0.9, 0.1)
    train = pa.table({"x": x, "label": (rng.random(n_rows) < rates).astype("int64")})
    sweep = np.linspace(0.0, 1.0, 60)
    test = pa.table({"x": sweep, "label": np.zeros(sweep.size, dtype="int64")})
    return InMemoryDatasetHandle(
        {"train": train, "test": test}, snapshot_id="sha256:u-shaped", label_column="label"
    )


def _sweep_predictions(adapter: Any, spec: ModelSpec, data: InMemoryDatasetHandle) -> np.ndarray:
    handle = _treated(data, spec)
    model = adapter.train(spec, handle, _ctx())
    predictions = adapter.predict(model, handle, "test")
    return predictions.column("prediction").to_numpy(zero_copy_only=False)


def _monotone_adapters() -> dict[str, Any]:
    from mbt_lightgbm.adapter import LightGBMTrainingAdapter
    from mbt_sklearn.adapter import SklearnTrainingAdapter
    from mbt_xgboost.adapter import XGBoostTrainingAdapter

    return {
        "xgboost": (
            XGBoostTrainingAdapter({}),
            {"max_depth": 4, "n_estimators": 60, "learning_rate": 0.2},
        ),
        "lightgbm": (
            LightGBMTrainingAdapter({}),
            {"num_leaves": 15, "n_estimators": 60, "learning_rate": 0.2, "min_child_samples": 5},
        ),
        "sklearn": (
            SklearnTrainingAdapter({}),
            {"estimator": "hist_gradient_boosting", "max_iter": 60, "learning_rate": 0.2},
        ),
    }


@pytest.mark.parametrize("name", sorted(_monotone_adapters()))
def test_monotone_constraint_bends_the_fitted_model(name: str) -> None:
    adapter, hyperparameters = _monotone_adapters()[name]
    data = _u_shaped_dataset()

    free = _sweep_predictions(adapter, _spec(name, {"include": ["*"]}, **hyperparameters), data)
    assert free.min() < free[-1] - 0.05, f"{name}: fixture is not non-monotone enough"
    assert np.diff(free).min() < -0.05, f"{name}: unconstrained fit should dip"

    constrained = _sweep_predictions(
        adapter,
        _spec(name, {"include": ["*"], "monotonic": {"x": "increasing"}}, **hyperparameters),
        data,
    )
    assert np.diff(constrained).min() >= -MONOTONE_TOLERANCE, name


def test_decreasing_is_the_mirror_of_increasing() -> None:
    from mbt_xgboost.adapter import XGBoostTrainingAdapter

    predictions = _sweep_predictions(
        XGBoostTrainingAdapter({}),
        _spec(
            "xgboost",
            {"include": ["*"], "monotonic": {"x": "decreasing"}},
            max_depth=4,
            n_estimators=60,
            learning_rate=0.2,
        ),
        _u_shaped_dataset(),
    )
    assert np.diff(predictions).max() <= MONOTONE_TOLERANCE


def test_the_inline_transforms_spelling_constrains_identically() -> None:
    """`transforms: {x: {monotonic: ...}}` and `monotonic: {x: ...}` are two
    spellings of one constraint, so they must produce the same model."""
    from mbt_xgboost.adapter import XGBoostTrainingAdapter

    adapter, data = XGBoostTrainingAdapter({}), _u_shaped_dataset()
    hyperparameters = {"max_depth": 4, "n_estimators": 60, "learning_rate": 0.2}
    standalone = _sweep_predictions(
        adapter,
        _spec("xgboost", {"include": ["*"], "monotonic": {"x": "increasing"}}, **hyperparameters),
        data,
    )
    inline = _sweep_predictions(
        adapter,
        _spec(
            "xgboost",
            {"include": ["*"], "transforms": {"x": {"monotonic": "increasing"}}},
            **hyperparameters,
        ),
        data,
    )
    assert standalone.tolist() == inline.tolist()


# -- declared categoricals -------------------------------------------------------


def _coded_categorical_dataset(
    n_rows: int = 1500, levels: int = 26, seed: int = 5
) -> InMemoryDatasetHandle:
    """The F24 high-cardinality shape, int-coded: positive rates ALTERNATE by
    code, so separating the levels needs unordered subset splits rather than
    one threshold per level. Read as a magnitude the column is nearly useless
    on a small budget; read as a category it is the whole signal."""
    rng = np.random.default_rng(seed)
    weights = np.array([1.0 / (i + 1) for i in range(levels)])
    weights /= weights.sum()
    code = rng.choice(levels, size=n_rows, p=weights)
    rates = np.where(code % 2 == 1, 0.9, 0.1)
    table = pa.table(
        {
            "contract_code": code.astype("int8"),
            "noise": rng.normal(size=n_rows),
            "label": (rng.random(n_rows) < rates).astype("int64"),
        }
    )
    cut = int(n_rows * 0.75)
    return InMemoryDatasetHandle(
        {"train": table.slice(0, cut), "test": table.slice(cut)},
        snapshot_id="sha256:coded-categorical",
        label_column="label",
    )


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_declaring_an_int_coded_column_categorical_recovers_the_signal(name: str) -> None:
    """The showcase's `wide_hooks.py` cast, now a spec field: without the
    declaration a shallow booster reads the code as a magnitude; with it, the
    levels split natively."""
    adapters = _monotone_adapters()
    adapter, _ = adapters[name]
    data = _coded_categorical_dataset()
    # A deliberately small budget of depth-1 stumps: read as a magnitude the
    # column needs roughly one threshold per level, which this cannot afford;
    # read as a category, one unordered subset split does it.
    hyperparameters = {"max_depth": 1, "n_estimators": 10, "learning_rate": 0.3}
    if name == "lightgbm":
        hyperparameters = {"num_leaves": 2, "n_estimators": 10, "learning_rate": 0.3}

    def auc(features: dict[str, Any]) -> float:
        spec = _spec(name, features, **hyperparameters)
        handle = _treated(data, spec)
        model = adapter.train(spec, handle, _ctx())
        return float(adapter.evaluate(model, handle, "test", ROC_AUC).metrics["roc_auc"])

    inferred = auc({"include": ["*"]})
    declared = auc({"include": ["*"], "categorical": ["contract_code"]})
    assert declared > inferred + 0.1, (name, inferred, declared)


def test_rare_levels_pool_into_other_and_catch_unseen_values() -> None:
    """`min_frequency` lives in the shared encoder because the level map is
    already persisted with the artifact - so the pooled bucket survives export,
    load, and a value the train split never contained."""
    from mbt_xgboost.adapter import XGBoostTrainingAdapter

    levels = ["north"] * 400 + ["south"] * 380 + ["yukon"] * 3 + ["nunavut"] * 2
    rng = np.random.default_rng(9)
    table = pa.table(
        {
            "region": levels,
            "label": rng.integers(0, 2, size=len(levels)).astype("int64"),
        }
    )
    unseen = pa.table({"region": ["north", "atlantis"], "label": np.array([0, 1], dtype="int64")})
    data = InMemoryDatasetHandle(
        {"train": table, "test": unseen}, snapshot_id="sha256:pool", label_column="label"
    )
    spec = _spec(
        "xgboost",
        {"include": ["*"], "categorical": {"region": {"min_frequency": 0.05}}},
        max_depth=2,
        n_estimators=5,
    )
    adapter = XGBoostTrainingAdapter({})
    handle = _treated(data, spec)
    model = adapter.train(spec, handle, _ctx())
    assert model.categories["region"] == ["north", "south", "__other__"]
    # A level absent from train scores as the pooled bucket, not as missing.
    assert adapter.predict(model, handle, "test").num_rows == 2
