#!/usr/bin/env python3
"""Evidently feature-stability gate for the wide cadence (SHOW-20).

Enforces that the features churn_wide_automl actually trains on stay
distributionally stable, with exit-code-2 semantics like every other mbt
quality verdict. Two phases cover the gap between training and serving:

- ``--phase train`` runs BEFORE promotion: reference is the train split,
  current is the test split of the newest complete materialization, both
  restricted to the committed selected features. A breach blocks
  ``mbt promote`` (the model's features were already unstable inside its
  own training window). On pass, ``--export-reference`` persists the
  reference frame; DAG task containers are ephemeral, so the serving gate
  needs a baseline that outlives ``target/``.
- ``--phase serving`` runs AFTER every ``mbt score``: reference is the
  exported baseline, current is the scoring batch. This checks stability
  through first deployment and every monthly batch after it.

Both phases render the DS-facing Evidently HTML report (the old
evidently_report.py output) alongside the verdict. The gate complements
mbt's OWN enforcing feature_shift / prediction_shift monitors (PSI/KS
against the champion's training-time baseline, ADR-21): those gate the
scoring run itself; this one adds Evidently's per-column drift tests and
the pre-promotion phase, and stays showcase-local by design (no mbt
package depends on evidently; the pin lives in the runner image).

Exit codes: 0 stable, 2 drift breach (quality verdict, deterministic - the
DAG routes it to AirflowFailException), 1 missing inputs or an
unrecognizable evidently payload.

Usage (from the project root):

    python scripts/evidently_gate.py --phase train \
        [--export-reference /workspace/monitoring/wide_reference.parquet]
    python scripts/evidently_gate.py --phase serving \
        [--reference <parquet>] [--current <parquet>]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

BEGIN = "# BEGIN selected-features"
END = "# END selected-features"
DEFAULT_REFERENCE = Path("/workspace/monitoring/wide_reference.parquet")
TRAIN_ROOT = Path("target/datasets/wide_churn_training")
SCORE_ROOT = Path("target/scoring_inputs/wide_retention_scoring")


def read_selected_features(model_file: Path) -> list[str]:
    """The committed include list between the markers - the single source of truth."""
    lines = model_file.read_text().splitlines()
    begin = next((i for i, line in enumerate(lines) if BEGIN in line), None)
    end = next((i for i, line in enumerate(lines) if END in line), None)
    if begin is None or end is None or end <= begin:
        sys.exit(f"error: {model_file} has no '{BEGIN}' ... '{END}' marker block")
    features = [
        line.strip().removeprefix("- ")
        for line in lines[begin + 1 : end]
        if line.strip().startswith("- ")
    ]
    if not features:
        sys.exit(
            f"error: empty selected-features block in {model_file}; "
            "run scripts/select_features.py first"
        )
    return features


def load_categorical_codes(model_file: Path) -> list[str]:
    hooks_file = model_file.resolve().parent / "wide_hooks.py"
    spec = importlib.util.spec_from_file_location("_gate_wide_hooks", hooks_file)
    if spec is None or spec.loader is None:
        sys.exit(f"error: cannot import {hooks_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.CATEGORICAL_CODES)


def newest_dir(root: Path, required: tuple[str, ...], hint: str, by: str = "mtime") -> Path:
    """Pick a complete dir by recency, or by="size" for the FULL build.

    Training materializations use size (sampled builds are strict subsets,
    so the largest first split is the full panel regardless of which build
    ran last); scoring runs use recency (the latest batch is the one to
    gate).
    """
    complete = [d for d in root.glob("*") if all((d / name).is_file() for name in required)]
    if not complete:
        sys.exit(f"error: no {'/'.join(required)} under {root}; {hint}")
    if by == "size":
        return max(complete, key=lambda d: ((d / required[0]).stat().st_size, d.stat().st_mtime))
    return max(complete, key=lambda d: d.stat().st_mtime)


def load_frame(path: Path, features: list[str], codes: list[str]) -> pd.DataFrame:
    """The frame restricted to the selected features, hook-cast applied.

    Casting the categorical codes on BOTH sides mirrors wide_hooks.py, so
    Evidently compares them as categories, exactly as the trainers see them.
    """
    frame = pd.read_parquet(path)
    present = [c for c in features if c in frame.columns]
    missing = sorted(set(features) - set(present))
    if missing:
        print(f"warning: {path} lacks selected features: {', '.join(missing)}", file=sys.stderr)
    frame = frame[present]
    for name in codes:
        if name in frame.columns:
            frame[name] = frame[name].astype(str)
    return frame


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> dict:
    """The only evidently-touching function (evidently lives in the runner image)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)
    # evidently 0.7 returns the snapshot; older versions mutate the report.
    snapshot = result if hasattr(result, "save_html") else report
    snapshot.save_html(str(out_html))
    return snapshot.dict()


def summarize(payload: dict) -> tuple[float, list[dict]]:
    """Drifted-column share + per-feature scores from a snapshot dict.

    An evidently 0.7.x snapshot entry looks like ``{"metric_name":
    "ValueDrift(column=age_years,method=...,threshold=0.1)", "config":
    {"type": "evidently:metric_v2:ValueDrift", "column": "age_years", ...},
    "value": 0.0116}`` (captured live from 0.7.20). Parsing leans on the
    typed config and falls back to the rendered name, tolerating shape
    drift in either direction.
    """
    share = None
    rows = []
    for entry in payload.get("metrics", []):
        config = entry.get("config") or {}
        kind = str(config.get("type") or entry.get("metric_name") or "")
        value = entry.get("value")
        if "DriftedColumnsCount" in kind:
            share = float(value["share"] if isinstance(value, dict) else value)
        elif "ValueDrift" in kind:
            column = config.get("column")
            if column is None:
                name = str(entry.get("metric_name", ""))
                column = name.split("column=", 1)[-1].split(",", maxsplit=1)[0].rstrip(")")
            score = value.get("drift_score", value) if isinstance(value, dict) else value
            rows.append({"feature": str(column), "score": float(score)})
    if share is None:
        sys.exit(
            "error: no DriftedColumnsCount metric in the evidently payload; "
            "the evidently version changed shape"
        )
    return share, sorted(rows, key=lambda r: (-r["score"], r["feature"]))


def gate_verdict(share: float, max_share: float) -> int:
    return 0 if share <= max_share else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=["train", "serving"], required=True)
    parser.add_argument("--model-file", type=Path, default=Path("models/churn_wide_automl.yml"))
    parser.add_argument(
        "--reference", type=Path, default=None, help="serving phase baseline parquet"
    )
    parser.add_argument("--current", type=Path, default=None, help="serving phase batch parquet")
    parser.add_argument(
        "--export-reference",
        type=Path,
        default=None,
        help="train phase: persist the baseline here on pass",
    )
    parser.add_argument("--max-drift-share", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=Path("drift_report.html"))
    args = parser.parse_args()

    features = read_selected_features(args.model_file)
    codes = load_categorical_codes(args.model_file)

    if args.phase == "train":
        split_dir = newest_dir(
            TRAIN_ROOT,
            ("train.parquet", "test.parquet"),
            "run `mbt build --select churn_wide_automl` first",
            by="size",
        )
        reference_path = split_dir / "train.parquet"
        current_path = split_dir / "test.parquet"
    else:
        reference_path = args.reference or DEFAULT_REFERENCE
        if not reference_path.is_file():
            if args.reference is not None:
                sys.exit(
                    f"error: no exported reference at {reference_path}; "
                    "run the train-phase gate first"
                )
            split_dir = newest_dir(
                TRAIN_ROOT,
                ("train.parquet",),
                "run the train-phase gate or `mbt build` first",
                by="size",
            )
            reference_path = split_dir / "train.parquet"
            print(
                f"warning: no exported reference; falling back to {reference_path}", file=sys.stderr
            )
        current_path = args.current
        if current_path is None:
            current_path = (
                newest_dir(SCORE_ROOT, ("score.parquet",), "run `mbt score` first")
                / "score.parquet"
            )
        elif not current_path.is_file():
            sys.exit(f"error: no scoring batch at {current_path}; run `mbt score` first")

    reference = load_frame(reference_path, features, codes)
    current = load_frame(current_path, features, codes)

    payload = run_drift(reference, current, args.out)
    share, rows = summarize(payload)

    print(f"reference: {reference_path}")
    print(f"current:   {current_path}")
    print(f"report:    {args.out}")
    for row in rows:
        print(f"  {row['feature']:<24} drift score {row['score']:.4f}")

    verdict = gate_verdict(share, args.max_drift_share)
    if verdict == 0 and args.phase == "train" and args.export_reference is not None:
        args.export_reference.parent.mkdir(parents=True, exist_ok=True)
        reference.to_parquet(args.export_reference)
        print(f"reference exported: {args.export_reference}")
    label, comparison = ("PASS", "<=") if verdict == 0 else ("BREACH", ">")
    print(
        f"{label}: drifted share {share:.2f} {comparison} "
        f"max {args.max_drift_share:.2f} (phase {args.phase})"
    )
    sys.exit(verdict)


if __name__ == "__main__":
    main()
