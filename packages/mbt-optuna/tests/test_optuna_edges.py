"""Optuna engine edges: every search-space dimension type, the all-pruned
failure mode, and the plugin descriptor (complements test_pruning)."""

from typing import Any

import pytest
from mbt_optuna.engine import OptunaTuningEngine

from mbt_adapter_base import CONTRACT_VERSION, TuningSpec


def test_suggest_covers_every_dimension_type() -> None:
    spec = TuningSpec.model_validate(
        {
            "engine": "optuna",
            "n_trials": 5,
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

    result = OptunaTuningEngine({}).tune(spec, objective, n_trials=5, seed=7)
    assert result.n_trials == 5 and result.n_pruned == 0
    assert set(result.best_params) == {"booster", "max_depth", "learning_rate", "subsample"}
    for params in proposals:
        assert params["booster"] in ("gbtree", "dart")
        assert isinstance(params["max_depth"], int) and 2 <= params["max_depth"] <= 6
        assert 1e-4 <= params["learning_rate"] <= 1e-1
        assert 0.5 <= params["subsample"] <= 1.0


def test_no_completed_trials_is_an_actionable_error() -> None:
    import optuna

    spec = TuningSpec.model_validate(
        {
            "engine": "optuna",
            "n_trials": 3,
            "search_space": {"quality": {"type": "uniform", "low": 0.0, "high": 1.0}},
            "objective": {"metric": "pr_auc", "direction": "maximize"},
        }
    )

    def objective(params: dict[str, Any]) -> float:
        raise optuna.TrialPruned()  # every trial aborts

    with pytest.raises(ValueError, match="no completed trials"):
        OptunaTuningEngine({}).tune(spec, objective, n_trials=3, seed=7)


def test_plugin_descriptor_wires_the_tuning_engine() -> None:
    from mbt_optuna.plugin import PLUGIN

    assert PLUGIN.name == "optuna"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.tuning is OptunaTuningEngine
    assert PLUGIN.fingerprint_packages == ["optuna"]
