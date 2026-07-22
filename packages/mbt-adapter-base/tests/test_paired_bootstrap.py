"""Paired-bootstrap delta unit tests (ADR-18): seeded and fully deterministic."""

import numpy as np
import pytest

from mbt_adapter_base.interchange import BootstrapDelta
from mbt_adapter_base.metrics import paired_bootstrap_delta
from mbt_adapter_base.specs import MetricSpec

AUC = MetricSpec(name="roc_auc", kind="builtin", greater_is_better=True)
BRIER = MetricSpec(name="brier", kind="builtin", greater_is_better=False)


def _sigmoid(latent: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-latent)))


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels plus a strong and a weak probabilistic scorer, deterministically.

    Scores are sigmoids of a noisy latent so class overlap is real and no
    model is degenerate (AUC roughly 0.98 strong, 0.76 weak)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.float64)
    sign = 2 * y - 1
    strong = _sigmoid(1.5 * sign + rng.standard_normal(n))
    weak = _sigmoid(0.5 * sign + rng.standard_normal(n))
    return y, strong, weak


def _bootstrap(spec: MetricSpec, y, challenger, champion, **kwargs) -> BootstrapDelta:
    defaults = {
        "greater_is_better": spec.greater_is_better,
        "confidence": 0.95,
        "n_resamples": 200,
        "seed": 7,
    }
    return paired_bootstrap_delta(spec, y, challenger, champion, **{**defaults, **kwargs})


def test_same_seed_reproduces_the_bound_exactly() -> None:
    y, strong, weak = _synthetic(300, seed=1)
    first = _bootstrap(AUC, y, strong, weak)
    second = _bootstrap(AUC, y, strong, weak)
    assert first == second
    assert first.n_resamples == 200


def test_clear_improvement_has_positive_lower_bound() -> None:
    y, strong, weak = _synthetic(1500, seed=2)
    result = _bootstrap(AUC, y, strong, weak)
    assert result.point > 0
    assert 0 < result.lower < result.point

    # lower-is-better metric: improvement is still oriented positive
    brier = _bootstrap(BRIER, y, strong, weak)
    assert brier.lower > 0


def test_noise_advantage_is_not_significant() -> None:
    """The P1 scenario: two equally good models, the challenger ahead on
    test-set noise alone. The point delta clears min_delta 0; the paired
    bootstrap refuses to call it an improvement."""
    rng = np.random.default_rng(3)
    n = 120
    y = (rng.random(n) < 0.3).astype(np.float64)
    sign = 2 * y - 1
    challenger = _sigmoid(0.8 * sign + rng.standard_normal(n))
    champion = _sigmoid(0.8 * sign + rng.standard_normal(n))
    result = _bootstrap(AUC, y, challenger, champion, n_resamples=500)
    assert result.point > 0.04  # the old criterion would promote this...
    assert result.lower < 0  # ...the new criterion blocks it


def test_identical_models_bound_is_zero() -> None:
    y, strong, _ = _synthetic(200, seed=4)
    result = _bootstrap(AUC, y, strong, strong)
    assert result.point == 0.0
    assert result.lower == 0.0  # still passes a min_delta 0.0 gate


def test_single_class_resamples_are_skipped() -> None:
    rng = np.random.default_rng(5)
    y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    challenger = np.clip(y * 0.8 + 0.1 * rng.random(6), 0.0, 1.0)
    champion = rng.random(6)
    result = _bootstrap(AUC, y, challenger, champion, n_resamples=100)
    assert 0 < result.n_resamples < 100  # resamples without the positive skipped


def test_all_degenerate_falls_back_to_point_delta() -> None:
    y = np.array([1.0])  # every resample is single-class
    result = _bootstrap(BRIER, y, np.array([0.9]), np.array([0.5]))
    assert result.n_resamples == 0
    assert result.point == pytest.approx(0.25 - 0.01)  # brier is lower-better
    assert result.lower == result.point


def test_bootstrap_metric_lower_bound_is_pessimistic_and_deterministic() -> None:
    """Single-model metric bound (R2-7): lower percentile for higher-is-better,
    upper for lower-is-better, and reproducible given the seed."""
    from mbt_adapter_base.metrics import bootstrap_metric_lower_bound, compute_metric

    y, strong, _ = _synthetic(500, seed=3)
    point = compute_metric(AUC, y, strong)
    lb = bootstrap_metric_lower_bound(AUC, y, strong, confidence=0.95, n_resamples=500, seed=1)
    assert lb < point  # higher-is-better -> the pessimistic bound is below the point
    again = bootstrap_metric_lower_bound(AUC, y, strong, confidence=0.95, n_resamples=500, seed=1)
    assert lb == again  # seeded -> deterministic

    b_point = compute_metric(BRIER, y, strong)
    ub = bootstrap_metric_lower_bound(BRIER, y, strong, confidence=0.95, n_resamples=500, seed=1)
    assert ub > b_point  # lower-is-better -> pessimistic bound is the UPPER percentile


def test_bootstrap_metric_lower_bound_all_degenerate_returns_point() -> None:
    from mbt_adapter_base.metrics import bootstrap_metric_lower_bound, compute_metric

    y = np.zeros(20)  # single class: every resample is skipped
    score = np.linspace(0.0, 1.0, 20)
    result = bootstrap_metric_lower_bound(BRIER, y, score, confidence=0.95, n_resamples=50, seed=1)
    assert result == compute_metric(BRIER, y, score)  # fell back to the point value
