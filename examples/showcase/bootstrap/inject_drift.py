"""Inject feature shift into the local scoring batch (demo/test asset).

Multiplies every numeric feature column in lake_local/scoring_batch by 3,
which blows well past the PSI 0.25 monitor threshold on the next
`mbt score` run: the scoring node goes monitor_failed (exit 2) and the
pushed mbt_shift_value breaches mbt_shift_threshold, firing the Prometheus
alert. Re-run sync_lake.py to restore clean data.
"""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH_DIR = Path(os.environ.get("SHOWCASE_LAKE_LOCAL", "/workspace/lake_local")) / "scoring_batch"
PROTECTED = {"user_id", "snapshot_date", "churned_90d", "upgraded_90d"}


def main() -> int:
    files = sorted(BATCH_DIR.glob("*.parquet"))
    if not files:
        print(f"no parquet under {BATCH_DIR}")
        return 1
    for path in files:
        table = pq.read_table(path)
        shifted = []
        for field in table.schema:
            column = table[field.name]
            if field.name not in PROTECTED and pa.types.is_floating(field.type):
                column = pc.multiply(column, 3.0)
            elif field.name not in PROTECTED and pa.types.is_integer(field.type):
                column = pc.multiply(column, 3)
            shifted.append(column)
        pq.write_table(pa.table(dict(zip(table.schema.names, shifted, strict=True))), path)
        print(f"injected 3x shift into numeric features of {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
