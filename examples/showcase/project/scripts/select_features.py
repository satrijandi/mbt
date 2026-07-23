#!/usr/bin/env python3
"""Select churn_wide_automl's features with the ds-helper funnel.

Reads the newest FULL-width train split that the probe build materialized
under ``target/datasets/wide_churn_training/`` and runs a four-stage
selection funnel over it:

1. drop columns that are almost entirely missing (> --missing-threshold);
2. drop single-value columns (no information);
3. drop the later column of every highly correlated numeric pair
   (> --corr-threshold, pairwise absolute Pearson);
4. a seeded LightGBM randomized search (StratifiedKFold, roc_auc,
   scale_pos_weight from the class balance) keeps features whose split
   importance in the best estimator is > 0, capped at --top-k.

The winners are rewritten between the ``BEGIN/END selected-features``
markers in ``models/churn_wide_automl.yml`` and the whole funnel is
documented in ``target/feature_selection_report.json`` (per-stage drops,
best params, CV score, final importances) for review alongside the diff.

The workflow is deliberately a committed diff, not a hidden training-time
step: the DS reruns this after retraining the probe, reviews which features
entered or left, and the PR shows exactly that. Editing the list flips the
model's config hash, so slim CI retrains exactly the AutoML model.

Reproducibility: ``--seed`` defaults to 42, deliberately the same seed the
probe and AutoML specs declare - one committed seed governs the probe
materialization, this search, and the final training. Rows are sorted by
(customer_id, inference_date) before splitting because materializations from
different data adapters may order rows differently, and the search is
single-threaded and LightGBM-deterministic so the committed list is
byte-identical across host and runner image.

Numeric-coded categoricals: CATEGORICAL_CODES is imported from
``models/wide_hooks.py`` (the same hook that casts them at train and
scoring time), so the funnel sees exactly the dtypes the trainers see.

Usage (from the project root, after `mbt build --select churn_wide_probe`):

    python scripts/select_features.py [--top-k 24]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROBE = "churn_wide_probe"
BEGIN = "# BEGIN selected-features"
END = "# END selected-features"

#: Row-identity and label columns, never feature candidates. inference_date
#: must be excluded here explicitly: mbt drops the split time column at
#: train time, but the raw materialized parquet still carries it.
NON_FEATURES = ["customer_id", "safe_id", "inference_date", "as_of_date", "is_churn"]
TARGET = "is_churn"

#: The ds-helper randomized-search grid (satrijandi/ds-helper
#: features_selection.py), minus min_data_in_leaf which is scale-bound below.
GRID = {
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "num_leaves": [6, 30, 50, 100],
    "max_depth": [3, 5, 6, 8],
    "min_child_weight": [1e-5, 1e-2, 1e-1, 1.0, 1e1, 1e2],
    "subsample": [0.1, 0.2, 0.5, 0.7, 0.9],
    "colsample_bytree": [0.1, 0.2, 0.5, 0.7, 0.9],
    "reg_alpha": [0.0, 1e-5, 1e-2, 1e-1, 1.0, 1e1, 1e2],
    "reg_lambda": [0.0, 1e-5, 1e-2, 1e-1, 1.0, 1e1, 1e2],
}
LEAF_LADDER = [20, 100, 200, 500, 2000]


def load_categorical_codes(model_file: Path) -> list[str]:
    """The DS-declared numeric-coded categoricals, from the shared hooks file."""
    hooks_file = model_file.resolve().parent / "wide_hooks.py"
    spec = importlib.util.spec_from_file_location("_selection_wide_hooks", hooks_file)
    if spec is None or spec.loader is None:
        sys.exit(f"error: cannot import {hooks_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.CATEGORICAL_CODES)


def load_excluded(model_file: Path) -> list[str]:
    """The model's `features.exclude` list: DS-declared ignored columns.

    The funnel must never offer these as candidates - entity ids, and
    columns the DS knows drift by construction (the wide models exclude
    `tenure_months`: it is anchored to calendar time, so the newest cohort
    always breaches a training-time PSI baseline no matter how predictive
    it looks inside the training window).
    """
    import yaml

    payload = yaml.safe_load(model_file.read_text())
    models = payload.get("models") or []
    if not models:
        sys.exit(f"error: no models in {model_file}")
    excluded = (models[0].get("features") or {}).get("exclude") or []
    return [str(name) for name in excluded]


def newest_materialization(root: Path, required: tuple[str, ...]) -> Path:
    """The FULL-build materialization with every required split file.

    Sampled-fraction builds materialize under their own keys and are strict
    SUBSETS of the full build (hash sampling), so the largest first split
    identifies the full panel, newest mtime breaking ties. Size, not
    recency: a DS's sampled what-if from the notebook must never silently
    become the selection input just because it ran last.
    """
    complete = [d for d in root.glob("*") if all((d / name).is_file() for name in required)]
    if not complete:
        sys.exit(
            f"error: no materialization under {root} with {', '.join(required)}; "
            f"run `mbt build --select {PROBE}` first"
        )
    return max(complete, key=lambda d: ((d / required[0]).stat().st_size, d.stat().st_mtime))


def load_frames(
    train_parquet: Path, categorical_codes: list[str], excluded: list[str] = ()
) -> tuple[pd.DataFrame, pd.Series]:
    """The candidate-feature frame and target, deterministically ordered."""
    df = pd.read_parquet(train_parquet)
    # mergesort is stable: different engines materialize different row
    # orders, and StratifiedKFold(shuffle=True) is index-sensitive.
    df = df.sort_values(["customer_id", "inference_date"], kind="mergesort").reset_index(drop=True)
    y = df[TARGET]
    dropped = [c for c in [*NON_FEATURES, *excluded] if c in df.columns]
    features = df.drop(columns=dropped)
    for name in categorical_codes:
        if name in features.columns:
            features[name] = features[name].astype(str)
    for name in features.columns:
        if features[name].dtype == object or pd.api.types.is_string_dtype(features[name]):
            features[name] = features[name].astype("category")
    return features, y


def remove_high_missing(features: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    missing = features.isnull().mean()
    dropped = [c for c in features.columns if missing[c] > threshold]
    return features.drop(columns=dropped), dropped


def remove_single_unique(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
    return features.drop(columns=dropped), dropped


def remove_correlated(features: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop the LATER column of every |corr| > threshold numeric pair."""
    numeric = features.select_dtypes(include="number")
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    return features.drop(columns=dropped), dropped


def select_features_lgbm(
    features: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int,
    seed: int,
    n_iter: int,
) -> tuple[list[tuple[str, int]], dict]:
    """ds-helper's select_features_lgbm: randomized search, keep importance > 0."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    # min_data_in_leaf is bounded by the frame: on the committed showcase
    # scale (~26k train rows) the raw ds-helper ladder up to 2000 combined
    # with subsample 0.1 yields near-empty trees and all-zero importances;
    # at the real 7M-row scale every rung participates.
    grid = dict(GRID)
    leaf_cap = max(20, int(0.02 * len(features)))
    grid["min_data_in_leaf"] = [v for v in LEAF_LADDER if v <= leaf_cap]

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    estimator = LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        # subsample only takes effect with a bagging frequency (the raw
        # ds-helper grid tunes a knob that is otherwise inert in sklearn).
        subsample_freq=1,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    search = RandomizedSearchCV(
        estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(features, y)

    importances = search.best_estimator_.feature_importances_
    ranked = sorted(zip(features.columns, importances, strict=True), key=lambda t: (-t[1], t[0]))
    selected = [(name, int(imp)) for name, imp in ranked if imp > 0]
    info = {
        "best_params": {
            k: float(v) if isinstance(v, float) else v for k, v in search.best_params_.items()
        },
        "best_cv_roc_auc": float(search.best_score_),
        "scale_pos_weight": scale_pos_weight,
        "zero_importance_dropped": [name for name, imp in ranked if imp == 0],
    }
    return selected, info


def probe_overlap(run_results: Path, selected: list[str], top_k: int) -> int | None:
    """Overlap with the probe's own gain ranking - a sanity signal, not a gate."""
    if not run_results.is_file():
        return None
    payload = json.loads(run_results.read_text())
    nodes = [r for r in payload["results"] if r.get("unique_id", "").endswith(f".{PROBE}")]
    importance: dict[str, float] = nodes[-1].get("feature_importance") or {} if nodes else {}
    if not importance:
        return None
    probe_top = {
        name for name, _ in sorted(importance.items(), key=lambda t: (-t[1], t[0]))[:top_k]
    }
    return len(probe_top & set(selected))


def rewrite_include(model_file: Path, features: list[str]) -> None:
    lines = model_file.read_text().splitlines(keepends=True)
    begin = next((i for i, line in enumerate(lines) if BEGIN in line), None)
    end = next((i for i, line in enumerate(lines) if END in line), None)
    if begin is None or end is None or end <= begin:
        sys.exit(f"error: {model_file} has no '{BEGIN}' ... '{END}' marker block")
    indent = lines[begin][: len(lines[begin]) - len(lines[begin].lstrip())]
    block = [f"{indent}include:\n"] + [f"{indent}  - {name}\n" for name in features]
    model_file.write_text("".join(lines[: begin + 1] + block + lines[end:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--train-parquet", type=Path, default=None)
    parser.add_argument("--model-file", type=Path, default=Path("models/churn_wide_automl.yml"))
    parser.add_argument("--report", type=Path, default=Path("target/feature_selection_report.json"))
    parser.add_argument("--run-results", type=Path, default=Path("target/run_results.json"))
    parser.add_argument("--seed", type=int, default=42, help="defaults to the wide specs' seed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--missing-threshold", type=float, default=0.95)
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    args = parser.parse_args()

    train_parquet = args.train_parquet
    if train_parquet is None:
        root = Path("target/datasets/wide_churn_training")
        train_parquet = (
            newest_materialization(root, ("train.parquet", "test.parquet")) / "train.parquet"
        )

    codes = load_categorical_codes(args.model_file)
    excluded = load_excluded(args.model_file)
    features, y = load_frames(train_parquet, codes, excluded)
    n_candidates = features.shape[1]

    features, high_missing = remove_high_missing(features, args.missing_threshold)
    features, single_unique = remove_single_unique(features)
    features, correlated = remove_correlated(features, args.corr_threshold)
    selected_ranked, lgbm_info = select_features_lgbm(
        features, y, folds=args.folds, seed=args.seed, n_iter=args.n_iter
    )
    selected_ranked = selected_ranked[: args.top_k]
    selected = sorted(name for name, _ in selected_ranked)

    overlap = probe_overlap(args.run_results, selected, args.top_k)
    report = {
        "train_parquet": str(train_parquet),
        "n_rows": len(y),
        "n_candidate_features": n_candidates,
        "excluded": excluded,
        "seed": args.seed,
        "top_k": args.top_k,
        "stages": {
            "high_missing": {"threshold": args.missing_threshold, "dropped": high_missing},
            "single_unique": {"dropped": single_unique},
            "correlated": {"threshold": args.corr_threshold, "dropped": correlated},
            "lgbm": lgbm_info,
        },
        "selected": [{"feature": name, "importance": imp} for name, imp in selected_ranked],
        "probe_top_k_overlap": overlap,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    rewrite_include(args.model_file, selected)
    print(
        f"funnel: {n_candidates} candidates"
        f" -{len(high_missing)} high-missing -{len(single_unique)} single-value"
        f" -{len(correlated)} correlated"
        f" -{len(lgbm_info['zero_importance_dropped'])} zero-importance"
    )
    print(f"selected {len(selected)} features into {args.model_file} (report: {args.report}):")
    for name in selected:
        print(f"  - {name}")
    if overlap is not None:
        print(f"probe gain top-{args.top_k} overlap: {overlap}/{len(selected)}")


if __name__ == "__main__":
    main()
