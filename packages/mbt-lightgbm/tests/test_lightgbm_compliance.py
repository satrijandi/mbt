"""mbt-lightgbm against the compliance suite: the extensibility proof (S8-02, G4).

Built using only public mbt-adapter-base contracts - zero mbt-core imports
anywhere in the package (verified by test_no_core_imports below).
"""

import subprocess
import sys
from typing import ClassVar

from mbt_lightgbm.adapter import LightGBMTrainingAdapter

from mbt_adapter_base.compliance import TrainingAdapterCompliance


class TestLightGBMCompliance(TrainingAdapterCompliance):
    adapter_factory = LightGBMTrainingAdapter
    plugin_module = "mbt_lightgbm.plugin"
    framework_modules = ("lightgbm",)
    valid_hyperparameters: ClassVar[dict] = {
        "num_leaves": 15,
        "n_estimators": 30,
        "learning_rate": 0.2,
    }
    auto_hyperparameter = "scale_pos_weight"


def test_no_core_imports() -> None:
    """G4: the adapter package must not touch mbt-core, only the contracts."""
    probe = (
        "import sys\n"
        "import mbt_lightgbm.plugin, mbt_lightgbm.adapter, mbt_lightgbm.params\n"
        "loaded = [m for m in sys.modules if m == 'mbt' or m.startswith('mbt.')]\n"
        "print(loaded)\n"
        "assert not loaded, f'mbt-core modules loaded: {loaded}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_threading_nondeterminism_warning() -> None:
    from mbt_adapter_base import EvaluationProtocol, EvaluationSpec, ModelSpec, TaskType

    spec = ModelSpec(
        name="m",
        task=TaskType.BINARY_CLASSIFICATION,
        adapter="lightgbm",
        owner="t@example.com",
        dataset="ref('d')",
        target="label",
        hyperparameters={"num_threads": 4},
        evaluation=EvaluationSpec(protocol=EvaluationProtocol(), metrics=["roc_auc"]),
        seed=5,
    )
    adapter = LightGBMTrainingAdapter({})
    warnings = adapter.nondeterminism_warnings(spec)
    assert warnings and "num_threads" in warnings[0]  # S8-04
    assert not adapter.nondeterminism_warnings(spec.model_copy(update={"hyperparameters": {}}))
