"""Post-hoc probability calibration (mbt_adapter_base/calibration.py, R2-8)."""

import numpy as np
import pytest

from mbt_adapter_base import ModelSpec
from mbt_adapter_base.calibration import Calibrator


def _two_level_batch() -> tuple[np.ndarray, np.ndarray]:
    """Scores that systematically over-predict: a 0.9-score group is really 50%
    positive and a 0.7-score group is really 30% positive."""
    scores = np.array([0.9] * 100 + [0.7] * 100, dtype=float)
    labels = np.array([1] * 50 + [0] * 50 + [1] * 30 + [0] * 70, dtype=float)
    return scores, labels


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_maps_inflated_scores_toward_empirical_rates(method: str) -> None:
    scores, labels = _two_level_batch()
    cal = Calibrator.fit(scores, labels, method)  # type: ignore[arg-type]
    t09, t07 = cal.transform(np.array([0.9])).item(), cal.transform(np.array([0.7])).item()
    # both methods recover the ~0.5 / ~0.3 empirical rates from the inflated scores
    assert t07 == pytest.approx(0.3, abs=0.1)
    assert t09 == pytest.approx(0.5, abs=0.1)
    assert t07 < t09  # monotonic: preserves rank order
    # every calibrated value is a probability
    out = cal.transform(np.linspace(0.0, 1.0, 20))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_sigmoid_calibration_smooths_a_small_separable_set() -> None:
    """Platt target smoothing (F17): on a small, nearly-separable calibration set
    the raw unregularized logistic overfits to a near-step function (mid-range
    scores forced to ~0/1); the smoothed fit keeps a moderate slope, so a score
    near the boundary maps near 0.5, not to an overconfident extreme."""
    scores = np.array([0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    cal = Calibrator.fit(scores, labels, "sigmoid")
    below, above = cal.transform(np.array([0.45, 0.55]))
    # smoothed, not a step function: the boundary scores stay near 0.5
    assert 0.3 < below < 0.5 < above < 0.7
    # a raw C=1e10 fit on the 0/1 labels gives a slope ~28 here; smoothing keeps it moderate
    assert cal.params["a"] < 15.0


def test_sigmoid_calibration_handles_a_single_class_split() -> None:
    """The duplicate-and-weight encoding presents both classes, so a calibration
    split that happens to be all-positive (or all-negative) is well-posed rather
    than raising the way a raw two-class LogisticRegression fit would (F17)."""
    scores = np.array([0.6, 0.7, 0.8, 0.9])
    labels = np.array([1, 1, 1, 1], dtype=float)  # all positive
    cal = Calibrator.fit(scores, labels, "sigmoid")
    out = cal.transform(scores)
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert np.all(out > 0.5)  # an all-positive set calibrates upward, not to a crash


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_json_roundtrip_is_identical(method: str) -> None:
    scores, labels = _two_level_batch()
    cal = Calibrator.fit(scores, labels, method)  # type: ignore[arg-type]
    reloaded = Calibrator.from_json(cal.to_json())
    assert reloaded.method == method
    probe = np.linspace(0.0, 1.0, 25)
    np.testing.assert_allclose(reloaded.transform(probe), cal.transform(probe))


def _model_spec(task: str, calibration: str) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "m",
            "task": task,
            "adapter": "xgboost",
            "owner": "t@example.com",
            "dataset": "ref('d')",
            "target": "y",
            "evaluation": {
                "protocol": {"split": "temporal"},
                "metrics": ["rmse"] if task == "regression" else ["roc_auc"],
            },
            "seed": 1,
            "calibration": calibration,
        }
    )


def test_calibration_rejected_on_regression_task() -> None:
    with pytest.raises(ValueError, match="binary_classification only"):
        _model_spec("regression", "isotonic")


def test_calibration_accepted_on_binary_task() -> None:
    spec = _model_spec("binary_classification", "sigmoid")
    assert spec.calibration == "sigmoid"
