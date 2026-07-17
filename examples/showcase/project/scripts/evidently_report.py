#!/usr/bin/env python3
"""Generate an Evidently drift report: training reference vs scoring batch.

A human-readable companion to mbt's OWN stability enforcement - the
feature_shift / prediction_shift monitors (PSI/KS against the champion's
training-time baseline, ADR-21) are what gate the pipeline with exit-code-2
semantics. This report is the DS-facing artifact: after `mbt build` and
`mbt score`, run it and open drift_report.html in JupyterLab to see the
per-feature train-vs-serving comparison Evidently renders.

Showcase-local by design: no mbt package depends on evidently; the pin
lives in the runner image.

Usage (from the project root):

    python scripts/evidently_report.py [--reference <parquet>] \
        [--current <parquet>] [--out drift_report.html]

Defaults discover the newest wide-cadence materializations under target/.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Columns that identify rows rather than describe them.
ID_COLUMNS = ["customer_id", "safe_id", "snapshot_date", "is_churn"]


def _newest(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        sys.exit(f"error: nothing matches {pattern!r}; run the build/score step first")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=None, help="training split parquet")
    parser.add_argument("--current", type=Path, default=None, help="scoring batch parquet")
    parser.add_argument("--out", type=Path, default=Path("drift_report.html"))
    args = parser.parse_args()

    reference = args.reference or _newest("target/datasets/wide_churn_training/*/train.parquet")
    current = args.current or _newest(
        "target/scoring_inputs/wide_retention_scoring/*/score.parquet"
    )

    ref = pd.read_parquet(reference).drop(columns=ID_COLUMNS, errors="ignore")
    cur = pd.read_parquet(current).drop(columns=ID_COLUMNS, errors="ignore")
    cur = cur[[c for c in cur.columns if c in ref.columns]]

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(current_data=cur, reference_data=ref)
    # evidently 0.7 returns the snapshot; older versions mutate the report.
    (result if hasattr(result, "save_html") else report).save_html(str(args.out))
    print(f"reference: {reference}")
    print(f"current:   {current}")
    print(f"report:    {args.out}")


if __name__ == "__main__":
    main()
