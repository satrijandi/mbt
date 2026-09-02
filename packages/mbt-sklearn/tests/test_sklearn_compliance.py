"""mbt-sklearn against the compliance suite.

Built using only public mbt-adapter-base contracts - zero mbt-core imports
anywhere in the package (verified by test_no_core_imports below), the same
bar mbt-lightgbm holds as the extensibility proof.

The suite runs three times, once per estimator, because "sklearn" is not one
model: a linear estimator and a tree ensemble differ in what they can express,
what they expose as importance, and how they handle the ordinal-code sentinel.
An adapter that only satisfied the contract for its default estimator would
pass a single run and fail the first user who changed one line of spec.
"""

import subprocess
import sys
from typing import ClassVar

import pytest
from mbt_sklearn.adapter import SUPPORTED_ESTIMATORS, SklearnTrainingAdapter

from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec, TaskType
from mbt_adapter_base.compliance import TrainingAdapterCompliance


class TestSklearnLogisticCompliance(TrainingAdapterCompliance):
    adapter_factory = SklearnTrainingAdapter
    plugin_module = "mbt_sklearn.plugin"
    framework_modules = ("sklearn",)
    valid_hyperparameters: ClassVar[dict] = {"estimator": "logistic", "C": 0.5, "max_iter": 200}
    # `logistic` is binary-only; the regression half of the suite exercises its
    # linear counterpart, so one class covers the linear family end to end.
    regression_hyperparameters: ClassVar[dict] = {"estimator": "linear", "alpha": 0.5}
    auto_hyperparameter = "class_weight"


class TestSklearnRandomForestCompliance(TrainingAdapterCompliance):
    adapter_factory = SklearnTrainingAdapter
    plugin_module = "mbt_sklearn.plugin"
    framework_modules = ("sklearn",)
    valid_hyperparameters: ClassVar[dict] = {
        "estimator": "random_forest",
        "n_estimators": 20,
        "max_depth": 4,
    }
    auto_hyperparameter = "class_weight"


class TestSklearnHistGradientBoostingCompliance(TrainingAdapterCompliance):
    adapter_factory = SklearnTrainingAdapter
    plugin_module = "mbt_sklearn.plugin"
    framework_modules = ("sklearn",)
    valid_hyperparameters: ClassVar[dict] = {
        "estimator": "hist_gradient_boosting",
        "max_iter": 30,
        "learning_rate": 0.2,
    }
    auto_hyperparameter = "class_weight"


def test_no_core_imports() -> None:
    """G4: the adapter package must not touch mbt-core, only the contracts."""
    probe = (
        "import sys\n"
        "import mbt_sklearn.plugin, mbt_sklearn.adapter, mbt_sklearn.params\n"
        "loaded = [m for m in sys.modules if m == 'mbt' or m.startswith('mbt.')]\n"
        "print(loaded)\n"
        "assert not loaded, f'mbt-core modules loaded: {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def _spec(**hyperparameters: object) -> ModelSpec:
    return ModelSpec(
        name="m",
        task=TaskType.BINARY_CLASSIFICATION,
        adapter="sklearn",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        hyperparameters=dict(hyperparameters),
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        seed=5,
    )


def test_threading_nondeterminism_warning() -> None:
    """The exact determinism tier is a claim about n_jobs=1; anything else warns."""
    adapter = SklearnTrainingAdapter({})
    warnings = adapter.nondeterminism_warnings(_spec(n_jobs=4))
    assert warnings and "n_jobs" in warnings[0]
    assert not adapter.nondeterminism_warnings(_spec())

    issues = adapter.validate(_spec(n_jobs=4))
    assert [i.severity for i in issues] == ["warning"]
    assert issues[0].field_path == "/hyperparameters/n_jobs"


def test_auto_resolves_class_weight_to_balanced() -> None:
    """sklearn spells imbalance correction as a strategy name, not xgboost's
    numeric scale_pos_weight, so '{{ auto }}' resolves to "balanced"."""
    from mbt_adapter_base import AUTO, DatasetProfile

    adapter = SklearnTrainingAdapter({})
    profile = DatasetProfile(
        n_rows={"train": 80, "test": 20},
        columns={"f1": "double", "label": "int64"},
        label_column="label",
        label_balance={"0": 0.8, "1": 0.2},
    )
    resolved = adapter.resolve_auto(_spec(class_weight=AUTO), profile)
    assert resolved.hyperparameters["class_weight"] == "balanced"


def test_auto_on_an_unsupported_hyperparameter_fails_loudly() -> None:
    from mbt_adapter_base import AUTO, DatasetProfile

    adapter = SklearnTrainingAdapter({})
    with pytest.raises(ValueError, match="only class_weight supports"):
        adapter.resolve_auto(
            _spec(C=AUTO),
            DatasetProfile(
                n_rows={"train": 8},
                columns={"label": "int64"},
                label_column="label",
            ),
        )


def test_every_advertised_estimator_is_reachable() -> None:
    """SUPPORTED_ESTIMATORS is what the README documents; it must match what
    param validation actually accepts, per task."""
    adapter = SklearnTrainingAdapter({})
    for task, names in SUPPORTED_ESTIMATORS.items():
        model = adapter.param_model(task)
        for name in names:
            assert model.model_validate({"estimator": name}) is not None
