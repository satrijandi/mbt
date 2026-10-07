"""Print the anchor the lake is at, derived from the data, never the clock.

The lake table's newest weekly cohort is its as-of date (bootstrap/
churn_panel.py writes one Monday cohort per week, and `advance` adds the
next). Every run anchors to that:

    build / score   the day after the newest cohort lands: the scoring
                    window ("7d") then holds exactly that cohort, and the
                    dataset's relative split windows end just before it
    --monitor       ten days later, past the 7d ground-truth maturity of a
                    run scored at the build anchor

So the Makefile, the Woodpecker pipelines, the Airflow DAGs, the CronJob and
the notebook all follow the lake when it moves a week, and a lake that has
not moved resolves to the same anchor every time - which keeps same-source
rebuilds byte-identical (generated_at == anchor, ADR-19).

Usage: python3 scripts/lake_anchor.py [--monitor] [--lake s3://mbt-lake/churn_panel]
Reads s3:// through boto3 and the AWS_* environment the stack provides; a
local directory works too.
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_LAKE = "s3://mbt-lake/churn_panel"
ANCHOR_DELAY = timedelta(days=1)
MONITOR_DELAY = timedelta(days=10)
COHORT_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})-[a-z]+-\d{3}\.parquet$")


def cohort_files(lake: str) -> list[str]:
    if not lake.startswith("s3://"):
        root = Path(lake)
        return [p.name for p in root.iterdir()] if root.is_dir() else []
    import boto3

    bucket, _, prefix = lake[len("s3://") :].partition("/")
    prefix = f"{prefix.rstrip('/')}/" if prefix else ""
    names: list[str] = []
    pages = (
        boto3.client("s3").get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    )
    for page in pages:
        names.extend(obj["Key"][len(prefix) :] for obj in page.get("Contents", []))
    return names


def as_of(lake: str) -> datetime:
    dates = [m.group(1) for name in cohort_files(lake) if (m := COHORT_FILE.match(name))]
    if not dates:
        raise SystemExit(f"{lake} holds no cohort files - seed the lake first (make seed)")
    return datetime.strptime(max(dates), "%Y-%m-%d")


def anchor(lake: str, *, monitor: bool = False) -> str:
    when = as_of(lake) + ANCHOR_DELAY + (MONITOR_DELAY if monitor else timedelta(0))
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lake", default=DEFAULT_LAKE)
    parser.add_argument("--monitor", action="store_true", help="the monitor anchor")
    args = parser.parse_args(argv)
    print(anchor(args.lake, monitor=args.monitor))
    return 0


if __name__ == "__main__":
    sys.exit(main())
