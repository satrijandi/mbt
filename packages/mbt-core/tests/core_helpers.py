"""Shared helpers for mbt-core tests (unique module name for pytest)."""

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

#: Fixed anchor used across compile tests (matches the fixture data range).
TEST_ANCHOR = datetime(2026, 7, 1, tzinfo=UTC)


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())
    return path


def write_subscriber_data(project_dir: Path, n_rows: int = 400) -> Path:
    """Deterministic sample parquet spanning ~200 days before TEST_ANCHOR."""
    base = TEST_ANCHOR.replace(tzinfo=None) - timedelta(days=200)
    rows = {
        "user_id": list(range(n_rows)),
        "snapshot_date": [base + timedelta(days=(i * 199) % 200) for i in range(n_rows)],
        "is_active": [i % 10 != 0 for i in range(n_rows)],
        "tenure_days": [30 + (i * 7) % 900 for i in range(n_rows)],
        "monthly_usage": [round((i * 13.7) % 500, 2) for i in range(n_rows)],
        "plan_type": [("basic", "pro", "enterprise")[i % 3] for i in range(n_rows)],
        "churned": [1 if (i * 31) % 100 < 22 else 0 for i in range(n_rows)],
    }
    out = project_dir / "data" / "subscribers"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "part-000.parquet"
    pq.write_table(pa.table(rows), path)
    return path
