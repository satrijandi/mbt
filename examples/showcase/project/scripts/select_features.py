#!/usr/bin/env python3
"""Distill the LightGBM probe's gain importance into a committed feature list.

Reads ``churn_wide_probe``'s ``feature_importance`` from
``target/run_results.json`` (exported by every training run) and rewrites
the include list between the ``BEGIN/END selected-features`` markers in
``models/churn_wide_automl.yml``.

The workflow is deliberately a committed diff, not a hidden training-time
step: the DS reruns this after retraining the probe, reviews which features
entered or left the top-K, and the PR shows exactly that. Editing the list
flips the model's config hash, so slim CI retrains exactly the AutoML model.

Usage (from the project root, after `mbt build --select churn_wide_probe`):

    python scripts/select_features.py [--top-k 24]
"""

import argparse
import json
import sys
from pathlib import Path

PROBE = "churn_wide_probe"
BEGIN = "# BEGIN selected-features"
END = "# END selected-features"


def selected_features(run_results: Path, top_k: int) -> list[str]:
    payload = json.loads(run_results.read_text())
    nodes = [r for r in payload["results"] if r.get("unique_id", "").endswith(f".{PROBE}")]
    if not nodes:
        sys.exit(f"error: no '{PROBE}' result in {run_results}; build the probe first")
    importance: dict[str, float] = nodes[-1].get("feature_importance") or {}
    if not importance:
        sys.exit(f"error: '{PROBE}' exported no feature importance in {run_results}")
    ranked = sorted(importance.items(), key=lambda item: (-item[1], item[0]))
    return sorted(name for name, _ in ranked[:top_k])


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
    parser.add_argument("--run-results", type=Path, default=Path("target/run_results.json"))
    parser.add_argument("--model-file", type=Path, default=Path("models/churn_wide_automl.yml"))
    args = parser.parse_args()

    features = selected_features(args.run_results, args.top_k)
    rewrite_include(args.model_file, features)
    print(f"selected {len(features)} features into {args.model_file}:")
    for name in features:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
