"""Hermetic units for the wide-cadence showcase scripts (SHOW-20).

The live tier (test_showcase_wide.py) proves the scripts against the real
stack; this module pins their pure logic in the fast suite: the ds-helper
selection funnel stages, the marker-block rewrite/read round-trip, the
evidently payload summarizer (against a canned snapshot dict - evidently
itself is a runner-image dep and must never be imported here), and the
shared categorical hook. Scripts are imported by file path (the pattern
test_bump_version.py established); everything writes under tmp_path only.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT = REPO_ROOT / "examples" / "showcase" / "project"


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_features = _import(PROJECT / "scripts" / "select_features.py", "wide_select_features")
evidently_gate = _import(PROJECT / "scripts" / "evidently_gate.py", "wide_evidently_gate")
wide_hooks = _import(PROJECT / "models" / "wide_hooks.py", "wide_hooks_module")


MODEL_YAML = """\
models:
  - name: churn_wide_automl
    features:
      # BEGIN selected-features
      include:
        - avg_session_min
        - contract_code
      # END selected-features
      exclude: [customer_id, safe_id]
"""


# -- funnel stages ----------------------------------------------------------


def test_remove_high_missing_drops_only_nearly_empty_columns() -> None:
    frame = pd.DataFrame(
        {
            "empty": [np.nan] * 99 + [1.0],
            "half": [np.nan] * 50 + [1.0] * 50,
            "full": np.arange(100.0),
        }
    )
    kept, dropped = select_features.remove_high_missing(frame, 0.95)
    assert dropped == ["empty"]
    assert list(kept.columns) == ["half", "full"]


def test_remove_single_unique_drops_constants_including_all_nan() -> None:
    frame = pd.DataFrame(
        {
            "constant": [7] * 10,
            "all_nan": [np.nan] * 10,
            "varied": list(range(10)),
        }
    )
    kept, dropped = select_features.remove_single_unique(frame)
    assert dropped == ["constant", "all_nan"]
    assert list(kept.columns) == ["varied"]


def test_remove_correlated_drops_the_later_column_of_a_pair() -> None:
    base = np.arange(100.0)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "original": base,
            "noise": rng.normal(size=100),
            "duplicate": base * 2.0 + 1.0,
            "category": pd.Series(["a", "b"] * 50, dtype="category"),
        }
    )
    kept, dropped = select_features.remove_correlated(frame, 0.9)
    assert dropped == ["duplicate"]
    assert list(kept.columns) == ["original", "noise", "category"]


def _signal_frame(rows: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    signal_num = rng.normal(size=rows)
    signal_cat = pd.Series(rng.choice(["keep", "leave"], rows), dtype="category")
    logits = 3.0 * signal_num + 2.0 * (signal_cat == "leave").to_numpy()
    y = pd.Series((rng.random(rows) < 1.0 / (1.0 + np.exp(-logits))).astype(int))
    frame = pd.DataFrame(
        {
            "signal_num": signal_num,
            "signal_cat": signal_cat,
            "noise_a": rng.normal(size=rows),
            "noise_b": rng.normal(size=rows),
            "noise_c": rng.normal(size=rows),
        }
    )
    return frame, y


def test_select_features_lgbm_finds_planted_signal_and_is_deterministic() -> None:
    frame, y = _signal_frame()
    first, info = select_features.select_features_lgbm(frame, y, folds=2, seed=42, n_iter=2)
    second, _ = select_features.select_features_lgbm(
        frame.copy(), y.copy(), folds=2, seed=42, n_iter=2
    )
    assert first == second
    names = [name for name, _ in first]
    assert "signal_num" in names
    assert "signal_cat" in names
    assert all(importance > 0 for _, importance in first)
    assert info["best_cv_roc_auc"] > 0.8
    assert set(info["zero_importance_dropped"]).isdisjoint({"signal_num", "signal_cat"})


def test_min_data_in_leaf_ladder_is_bounded_by_frame_size() -> None:
    assert select_features.LEAF_LADDER == [20, 100, 200, 500, 2000]
    # the clip rule: rungs above 2% of the rows drop out (26k rows -> 520 cap)
    cap = max(20, int(0.02 * 26000))
    assert [v for v in select_features.LEAF_LADDER if v <= cap] == [20, 100, 200, 500]


def test_load_frames_is_row_order_invariant(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(50),
            "safe_id": [f"sf-{i}" for i in range(50)],
            "snapshot_date": pd.Timestamp("2026-01-01"),
            "is_churn": rng.integers(0, 2, 50),
            "contract_code": rng.integers(0, 4, 50).astype(np.int8),
            "value": rng.normal(size=50),
        }
    )
    ordered, shuffled = tmp_path / "a.parquet", tmp_path / "b.parquet"
    frame.to_parquet(ordered)
    frame.sample(frac=1.0, random_state=1).to_parquet(shuffled)

    features_a, y_a = select_features.load_frames(ordered, ["contract_code"])
    features_b, y_b = select_features.load_frames(shuffled, ["contract_code"])
    pd.testing.assert_frame_equal(features_a, features_b)
    pd.testing.assert_series_equal(y_a, y_b)
    assert list(features_a.columns) == ["contract_code", "value"]
    assert features_a["contract_code"].dtype == "category"


def test_load_excluded_reads_the_ds_ignored_columns(tmp_path: Path) -> None:
    model_file = tmp_path / "model.yml"
    model_file.write_text(MODEL_YAML)
    assert select_features.load_excluded(model_file) == ["customer_id", "safe_id"]


def test_load_frames_honors_ds_excluded_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "snapshot_date": pd.Timestamp("2026-01-01"),
            "is_churn": [0, 1],
            "tenure_months": [12, 40],
            "value": [0.5, 0.7],
        }
    )
    path = tmp_path / "train.parquet"
    frame.to_parquet(path)
    features, _ = select_features.load_frames(path, [], ["tenure_months"])
    assert list(features.columns) == ["value"]


def test_newest_materialization_requires_complete_split_dirs(tmp_path: Path) -> None:
    partial, complete = tmp_path / "k1", tmp_path / "k2"
    partial.mkdir()
    (partial / "train.parquet").touch()
    complete.mkdir()
    (complete / "train.parquet").touch()
    (complete / "test.parquet").touch()
    picked = select_features.newest_materialization(tmp_path, ("train.parquet", "test.parquet"))
    assert picked == complete
    with pytest.raises(SystemExit):
        select_features.newest_materialization(tmp_path / "absent", ("train.parquet",))


# -- marker block round-trip ------------------------------------------------


def test_rewrite_include_round_trips_through_read(tmp_path: Path) -> None:
    model_file = tmp_path / "model.yml"
    model_file.write_text(MODEL_YAML)
    assert evidently_gate.read_selected_features(model_file) == ["avg_session_min", "contract_code"]

    select_features.rewrite_include(model_file, ["alpha", "beta", "gamma"])
    assert evidently_gate.read_selected_features(model_file) == ["alpha", "beta", "gamma"]
    # everything outside the markers is untouched
    text = model_file.read_text()
    assert "exclude: [customer_id, safe_id]" in text
    assert text.count("include:") == 1


def test_marker_block_must_exist_and_be_nonempty(tmp_path: Path) -> None:
    no_markers = tmp_path / "plain.yml"
    no_markers.write_text("models:\n  - name: x\n")
    with pytest.raises(SystemExit):
        select_features.rewrite_include(no_markers, ["a"])
    with pytest.raises(SystemExit):
        evidently_gate.read_selected_features(no_markers)

    empty_block = tmp_path / "empty.yml"
    empty_block.write_text(
        "features:\n  # BEGIN selected-features\n  include:\n  # END selected-features\n"
    )
    with pytest.raises(SystemExit):
        evidently_gate.read_selected_features(empty_block)


# -- evidently gate logic (no evidently import anywhere here) ---------------


def _snapshot_payload(share: float) -> dict:
    """The evidently 0.7.20 snapshot.dict() shape, captured live in the
    runner image (entry ids shortened; value shapes exact)."""
    return {
        "metrics": [
            {
                "id": "15e89f895b482f9b84ba7274ed18a106",
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {
                    "type": "evidently:metric_v2:DriftedColumnsCount",
                    "drift_share": 0.5,
                },
                "value": {"count": 2.0, "share": share},
            },
            {
                "id": "3acf044ff6c9f4748372a97e2c5994d8",
                "metric_name": (
                    "ValueDrift(column=avg_session_min,"
                    "method=Wasserstein distance (normed),threshold=0.1)"
                ),
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "avg_session_min",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.001,
            },
            {
                "id": "9b6d1f0a52a44de0a8b7b5b8c9f01234",
                "metric_name": (
                    "ValueDrift(column=contract_code,method=chi-square p_value,threshold=0.05)"
                ),
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "contract_code",
                },
                "value": 0.74,
            },
        ],
        "tests": [],
    }


def test_summarize_extracts_share_and_ranked_feature_scores() -> None:
    share, rows = evidently_gate.summarize(_snapshot_payload(0.4))
    assert share == 0.4
    assert [r["feature"] for r in rows] == ["contract_code", "avg_session_min"]
    assert rows[0]["score"] == 0.74


def test_summarize_falls_back_to_the_rendered_metric_name() -> None:
    payload = _snapshot_payload(0.1)
    for entry in payload["metrics"]:
        entry["config"] = {}
    share, rows = evidently_gate.summarize(payload)
    assert share == 0.1
    assert {r["feature"] for r in rows} == {"avg_session_min", "contract_code"}


def test_summarize_rejects_unrecognized_payloads() -> None:
    with pytest.raises(SystemExit):
        evidently_gate.summarize({"metrics": [{"metric_name": "SomethingElse", "value": 1}]})


def test_gate_verdict_uses_exit_code_two_for_quality() -> None:
    assert evidently_gate.gate_verdict(0.3, 0.3) == 0
    assert evidently_gate.gate_verdict(0.31, 0.3) == 2


def test_gate_load_frame_restricts_and_casts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "contract_code": np.array([0, 3], dtype=np.int8),
            "avg_session_min": [9.5, 8.1],
        }
    )
    path = tmp_path / "batch.parquet"
    frame.to_parquet(path)
    loaded = evidently_gate.load_frame(
        path, ["contract_code", "avg_session_min", "not_there"], ["contract_code"]
    )
    assert list(loaded.columns) == ["contract_code", "avg_session_min"]
    assert loaded["contract_code"].tolist() == ["0", "3"]


# -- the shared categorical hook --------------------------------------------


def test_wide_hooks_casts_declared_codes_and_preserves_the_rest() -> None:
    table = pa.table(
        {
            "contract_code": pa.array([0, 1, 3], type=pa.int8()),
            "is_churn": pa.array([0, 1, 0]),
            "avg_session_min": pa.array([9.5, 8.1, 7.7]),
        }
    )
    out = wide_hooks.transform_features(table, ctx=None)
    assert out.schema.field("contract_code").type == pa.string()
    assert out.column("contract_code").to_pylist() == ["0", "1", "3"]
    assert out.column("is_churn").to_pylist() == [0, 1, 0]
    assert out.schema.field("avg_session_min").type == pa.float64()

    untouched = pa.table({"other": pa.array([1, 2])})
    assert wide_hooks.transform_features(untouched, ctx=None) is untouched


def test_selection_report_shape_matches_what_the_live_test_reads(tmp_path: Path) -> None:
    """The report keys the live tier asserts on, pinned hermetically."""
    frame, y = _signal_frame(120)
    selected, info = select_features.select_features_lgbm(frame, y, folds=2, seed=42, n_iter=1)
    report = {
        "stages": {"lgbm": info},
        "selected": [{"feature": name, "importance": imp} for name, imp in selected],
    }
    encoded = json.loads(json.dumps(report))
    assert encoded["stages"]["lgbm"]["best_params"]
    assert all(row["importance"] > 0 for row in encoded["selected"])
