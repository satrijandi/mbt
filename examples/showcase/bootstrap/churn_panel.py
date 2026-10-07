"""The showcase's ONE lake table: churn_panel (runs inside the runner image).

Everything the showcase trains, scores, and monitors on lives in a single
wide table in the lake, the way a gold-layer customer snapshot usually does:

    s3://mbt-lake/churn_panel/<inference_date>-<state>-<chunk>.parquet

One row per customer per weekly ``inference_date``, every Monday from
2026-08-03 to the lake's as-of date - 2026-09-28 (9 cohorts) as seeded, one
week later per ``advance`` - carrying

    customer_id, inference_date   the natural key
    is_active                     the population flag: churned customers keep
                                  their rows, marked inactive, so specs select
                                  the population with a filter
    16 named features             demographic, login, and transaction signal
    f0000 .. fNNNN                pure-noise columns - the "huge" knob; a real
                                  lake table carries hundreds of columns no
                                  model uses
    is_churn                      churned during the 7 days after
                                  inference_date; NULL where there is no
                                  outcome to know: inactive rows, and any
                                  cohort whose outcome week has not closed by
                                  the as-of date - which is always the newest
                                  one, the as-of date itself

Nothing about it is committed: ``seed`` synthesizes it deterministically
(fixed seed) straight into the lake. Scale is a knob, not a fork -
``--scale huge`` is ~1.1M rows x ~320 columns through exactly the same code,
generated and written one chunk at a time so memory stays bounded.

The lake's clock is its newest cohort: the as-of date IS the newest
``inference_date`` in the listing, so the table carries its own time and every
anchor derives from it (project/scripts/lake_anchor.py). The newest cohort is
the scoring batch, and only it changes after it lands. Each change rewrites
that cohort under NEW file names and then deletes the old ones, never in
place: Spark pins an object-store source by its file LISTING, so an in-place
rewrite would change the data under a pinned manifest without changing its
snapshot.

    seed           write the cohorts up to 2026-09-28; the ones whose outcome
                   week is still open on that date "open" (label NULL), and
                   remove any later cohort an advance added
    land-outcomes  the newest cohort's outcome week closed: rewrite it with
                   its labels
    inject-drift   rewrite the newest cohort with numeric features x3, which
                   breaches the scoring monitors' PSI threshold
    reset          rewrite the newest cohort "open" again (undoes both)
    advance        a week passes: the newest cohort's outcomes land and the
                   next Monday's cohort arrives, open. It refuses to pass the
                   real calendar unless told to simulate the future, because a
                   cohort dated after today describes a population that does
                   not exist yet

``reset`` restores the exact seeded file names and bytes, so the source
snapshot pinned before a change is valid again afterwards. The simulation is
one sequential random stream, so an advanced cohort is the same whenever and
however often it is generated, and advancing never changes an earlier one.

Buckets are created WITHOUT any TTL/retention configuration on purpose: mbt
clean refuses s3:// artifact stores and nothing protects champion objects
server-side, so a retention rule would silently break champion gates,
evaluation, and scoring (DESIGN.md).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LAKE_BUCKET = "mbt-lake"
ARTIFACT_BUCKET = "mbt-artifacts"
TABLE = "churn_panel"
DEFAULT_LAKE = f"s3://{LAKE_BUCKET}/{TABLE}"

SEED = 20260717
FIRST_COHORT = datetime(2026, 8, 3)
CADENCE = timedelta(weeks=1)
#: The day ``seed`` writes the lake as of, and the label's outcome window (the
#: dataset's label.horizon). A cohort's is_churn is known only once its whole
#: outcome week has passed by the as-of date: 2026-09-21's closed on
#: 2026-09-28, so it is labelled; 2026-09-28's closes on 2026-10-05, so it is
#: open. ``advance`` moves the lake's as-of date a week at a time.
AS_OF = datetime(2026, 9, 28)
LABEL_HORIZON = timedelta(days=7)


def cohorts_until(as_of: datetime) -> list[datetime]:
    """Every Monday cohort from the first up to and including ``as_of``. There
    is never a later one: that population does not exist yet on the as-of
    date."""
    weeks = (as_of - FIRST_COHORT) // CADENCE
    return [FIRST_COHORT + week * CADENCE for week in range(weeks + 1)]


#: The cohorts as seeded: every Monday from 2026-08-03 to 2026-09-28.
COHORTS = cohorts_until(AS_OF)
NEWEST = COHORTS[-1]

#: name -> (customers, noise columns). The default keeps the live tier fast;
#: huge is the biggest shape the stack is sized for - SeaweedFS's capacity is
#: pinned at 6.4GB (compose/docker-compose.yml) and the artifact bucket shares
#: it. `seed --customers/--noise-columns` goes past it on a bigger lake.
SCALES = {"default": (3000, 48), "huge": (100_000, 300)}
ROWS_PER_FILE = 100_000

#: The states a cohort's files can be written in. A cohort starts matured or
#: open by `outcome_known`; land-outcomes / inject-drift / reset rewrite the
#: newest one.
STATES = ("open", "matured", "drifted")
#: Never shifted by inject-drift: keys, the population flag, the label, and the
#: numeric-coded categorical (tripling it would invent unseen levels, not shift).
UNSHIFTED = {"customer_id", "inference_date", "is_active", "is_churn", "contract_code"}
META_KEY = b"churn_panel"


def outcome_known(when: datetime, as_of: datetime = AS_OF) -> bool:
    """Whether a cohort's outcome week has closed by the as-of date."""
    return when + LABEL_HORIZON <= as_of


REGIONS = np.array(["north", "south", "east", "west", "central"])
INCOME_BANDS = np.array(["low", "mid", "upper_mid", "high"])
PLAN_TIERS = np.array(["basic", "plus", "premium"])
TOP_CATEGORIES = np.array(["grocery", "travel", "dining", "utilities", "retail", "other"])

#: The named (non-noise) feature columns, in table order. contract_code is
#: numeric-coded (0 = month-to-month ... 3 = two-year) with a deliberately
#: NON-monotone churn effect, so a model that reads it as a magnitude loses
#: signal - the models declare it under features.categorical.
FEATURES = (
    "age_years",
    "region",
    "income_band",
    "plan_tier",
    "contract_code",
    "household_size",
    "login_days_30d",
    "days_since_login",
    "sessions_30d",
    "avg_session_min",
    "txn_cnt_30d",
    "txn_amt_sum_30d",
    "txn_amt_avg_90d",
    "merchant_diversity",
    "top_category",
    "support_tickets_90d",
)


@dataclass(frozen=True)
class Params:
    customers: int
    noise_columns: int
    rows_per_file: int = ROWS_PER_FILE

    def to_json(self) -> str:
        return json.dumps(
            {
                "customers": self.customers,
                "noise_columns": self.noise_columns,
                "rows_per_file": self.rows_per_file,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> Params:
        return cls(**json.loads(raw))


def schema(noise_columns: int) -> pa.Schema:
    return pa.schema(
        [
            ("customer_id", pa.int64()),
            ("inference_date", pa.timestamp("us")),
            ("is_active", pa.bool_()),
            ("age_years", pa.int64()),
            ("region", pa.string()),
            ("income_band", pa.string()),
            ("plan_tier", pa.string()),
            ("contract_code", pa.int8()),
            ("household_size", pa.int64()),
            ("login_days_30d", pa.float64()),
            ("days_since_login", pa.float64()),
            ("sessions_30d", pa.float64()),
            ("avg_session_min", pa.float64()),
            ("txn_cnt_30d", pa.float64()),
            ("txn_amt_sum_30d", pa.float64()),
            ("txn_amt_avg_90d", pa.float64()),
            ("merchant_diversity", pa.float64()),
            ("top_category", pa.string()),
            ("support_tickets_90d", pa.int64()),
            *((f"f{i:04d}", pa.float32()) for i in range(noise_columns)),
            ("is_churn", pa.int64()),
        ]
    )


# -- simulation -----------------------------------------------------------------


@dataclass
class Cohort:
    index: int
    when: datetime
    #: the table's signal columns (everything but the noise), one row per
    #: customer in the book this week
    columns: dict[str, np.ndarray]
    #: realized outcome for every row; meaningful only where is_active
    churned: np.ndarray


def _draw_customers(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Static attributes for ``n`` customers, in the stream's fixed order."""
    return {
        "age_years": rng.integers(18, 75, n),
        "region": REGIONS[rng.integers(0, len(REGIONS), n)],
        "income_band": INCOME_BANDS[rng.integers(0, len(INCOME_BANDS), n)],
        "plan_tier": PLAN_TIERS[rng.integers(0, len(PLAN_TIERS), n)],
        "top_category": TOP_CATEGORIES[rng.integers(0, len(TOP_CATEGORIES), n)],
        "contract_code": rng.integers(0, 4, n).astype(np.int8),
        "household_size": rng.integers(1, 6, n),
        "login_base": np.clip(rng.normal(16.0, 7.0, n), 0.5, 30.0),
        "txn_base": np.clip(rng.normal(28.0, 14.0, n), 1.0, 120.0),
    }


def simulate(customers: int, until: datetime = AS_OF) -> Iterator[Cohort]:
    """Yield the weekly cohorts up to ``until`` in order, deterministically.

    Replaying the whole simulation is how one cohort is rebuilt on its own:
    the signal columns are cheap (a handful of vectors per week) and the
    random stream is a pure function of the seed, so every replay agrees -
    and a replay to a later ``until`` yields the earlier cohorts unchanged.
    """
    rng = np.random.default_rng(SEED)
    # The pool is bigger than the starting base: every week ~6% fresh
    # customers join while churners leave. Without acquisition, 10 weeks of
    # survivor culling would leave only low-hazard customers and the newest
    # cohort would carry almost no discriminable signal.
    pool = customers * 2
    attributes = _draw_customers(rng, pool)

    in_book = np.zeros(pool, dtype=bool)
    active = np.zeros(pool, dtype=bool)
    in_book[:customers] = active[:customers] = True
    next_joiner = customers
    joiners_per_week = int(customers * 0.06)
    for index, when in enumerate(cohorts_until(until)):
        if index > 0:
            if next_joiner + joiners_per_week > in_book.size:
                # Advanced past what the seeded pool covers (about 16 weeks):
                # grow it from a stream of its own, so the main stream - and
                # with it every cohort generated so far - stays untouched.
                extra = _draw_customers(np.random.default_rng([SEED, in_book.size]), customers)
                attributes = {
                    name: np.concatenate([column, extra[name]])
                    for name, column in attributes.items()
                }
                in_book = np.concatenate([in_book, np.zeros(customers, dtype=bool)])
                active = np.concatenate([active, np.zeros(customers, dtype=bool)])
            fresh = slice(next_joiner, next_joiner + joiners_per_week)
            in_book[fresh] = active[fresh] = True
            next_joiner += joiners_per_week
        customer_id = np.arange(in_book.size, dtype=np.int64)
        age_years = attributes["age_years"]
        region = attributes["region"]
        income_band = attributes["income_band"]
        plan_tier = attributes["plan_tier"]
        top_category = attributes["top_category"]
        contract_code = attributes["contract_code"]
        household_size = attributes["household_size"]
        login_base = attributes["login_base"]
        txn_base = attributes["txn_base"]
        idx = np.flatnonzero(in_book)
        live = active[idx]
        n = idx.size

        # This week's snapshot of each customer's rolling activity. An inactive
        # (churned) customer's row reports no activity; the population filter
        # drops it.
        login_days = np.clip(login_base[idx] * rng.normal(1.0, 0.28, n), 0.0, 30.0).round(1)
        days_since_login = np.clip(rng.exponential(30.0 / (1.0 + login_days)), 0.0, 30.0).round(1)
        txn_cnt = np.clip(txn_base[idx] * rng.normal(1.0, 0.22, n), 0.0, None).round(0)
        txn_amt_sum = (txn_cnt * np.clip(rng.normal(42.0, 15.0, n), 5.0, None)).round(2)
        sessions = (login_days * np.clip(rng.normal(2.2, 0.6, n), 0.5, None)).round(0)
        session_min = np.clip(rng.normal(9.0, 3.5, n), 0.5, None).round(1)
        txn_avg = np.clip(rng.normal(40.0, 12.0, n), 1.0, None).round(2)
        diversity = np.clip((txn_cnt / 4.0) + rng.normal(0, 1.5, n), 1.0, None).round(0)
        tickets = rng.poisson(0.6 + 1.2 * (login_days < 8.0), n)
        for column in (login_days, txn_cnt, txn_amt_sum, sessions, session_min, txn_avg):
            column[~live] = 0.0
        days_since_login[~live] = 30.0
        diversity[~live] = 0.0
        tickets[~live] = 0

        # Weekly churn hazard: inactivity dominates, income adds a mild tilt, and
        # contract_code's effect is non-monotone (0 churns most, 3 second-most).
        hazard = (
            0.015
            + 0.30 * (login_days < 8.0)
            + 0.15 * (days_since_login > 10.0)
            + 0.24 * (txn_cnt < 14.0)
            + 0.03 * (income_band[idx] == "low")
            + 0.12 * (contract_code[idx] == 0)
            + 0.06 * (contract_code[idx] == 3)
        )
        churned = (rng.random(n) < hazard) & live

        yield Cohort(
            index=index,
            when=when,
            columns={
                "customer_id": customer_id[idx],
                "inference_date": np.full(n, when, dtype="datetime64[us]"),
                "is_active": live,
                "age_years": age_years[idx],
                "region": region[idx],
                "income_band": income_band[idx],
                "plan_tier": plan_tier[idx],
                "contract_code": contract_code[idx],
                "household_size": household_size[idx],
                "login_days_30d": login_days,
                "days_since_login": days_since_login,
                "sessions_30d": sessions,
                "avg_session_min": session_min,
                "txn_cnt_30d": txn_cnt,
                "txn_amt_sum_30d": txn_amt_sum,
                "txn_amt_avg_90d": txn_avg,
                "merchant_diversity": diversity,
                "top_category": top_category[idx],
                "support_tickets_90d": tickets,
            },
            churned=churned,
        )
        active[idx[churned]] = False


def cohort_tables(cohort: Cohort, params: Params, state: str) -> Iterator[pa.Table]:
    """The cohort's rows in ``state``, as chunks of at most rows_per_file.

    Noise is drawn per (cohort, chunk) from its own seeded stream, so any one
    chunk can be regenerated without replaying the noise of the others.
    """
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}")
    target = schema(params.noise_columns)
    live = cohort.columns["is_active"]
    labelled = state == "matured"
    label = pa.array(
        cohort.churned.astype(np.int64),
        type=pa.int64(),
        mask=~live if labelled else np.ones(live.size, dtype=bool),
    )
    n = live.size
    for chunk, start in enumerate(range(0, max(n, 1), params.rows_per_file)):
        stop = min(start + params.rows_per_file, n)
        rows = stop - start
        noise_rng = np.random.default_rng([SEED, cohort.index, chunk])
        noise = noise_rng.normal(0.0, 1.0, (params.noise_columns, rows)).astype(np.float32)
        arrays: dict[str, Any] = {k: v[start:stop] for k, v in cohort.columns.items()}
        arrays.update({f"f{i:04d}": noise[i] for i in range(params.noise_columns)})
        arrays["is_churn"] = label.slice(start, rows)
        table = pa.table(arrays, schema=target)
        if state == "drifted":
            table = _shift(table)
        yield table.replace_schema_metadata({META_KEY: params.to_json().encode()})


def _shift(table: pa.Table) -> pa.Table:
    columns = []
    for field in table.schema:
        column = table[field.name]
        if field.name not in UNSHIFTED and pa.types.is_floating(field.type):
            column = pc.multiply(column, pa.scalar(3.0, field.type))
        elif field.name not in UNSHIFTED and pa.types.is_integer(field.type):
            column = pc.multiply(column, pa.scalar(3, field.type))
        columns.append(column)
    return pa.table(columns, schema=table.schema)


def file_name(when: datetime, state: str, chunk: int) -> str:
    return f"{when:%Y-%m-%d}-{state}-{chunk:03d}.parquet"


# -- the lake: a local directory or an s3:// prefix -----------------------------


class Lake:
    """Where the table's files live: ``s3://bucket/prefix`` or a local dir."""

    def __init__(self, location: str) -> None:
        self.location = location.rstrip("/")
        self._s3: Any = None
        if self.location.startswith("s3://"):
            import boto3

            bucket, _, prefix = self.location[len("s3://") :].partition("/")
            self.bucket, self.prefix = bucket, f"{prefix}/" if prefix else ""
            self._s3 = boto3.client("s3")
        else:
            self.root = Path(self.location)

    def ensure_buckets(self) -> None:
        if self._s3 is None:
            self.root.mkdir(parents=True, exist_ok=True)
            return
        existing = {b["Name"] for b in self._s3.list_buckets().get("Buckets", [])}
        for bucket in (LAKE_BUCKET, ARTIFACT_BUCKET):
            if bucket not in existing:
                self._s3.create_bucket(Bucket=bucket)
                print(f"created bucket {bucket}")

    def names(self) -> list[str]:
        if self._s3 is None:
            return sorted(p.name for p in self.root.glob("*.parquet")) if self.root.is_dir() else []
        found: list[str] = []
        for page in self._s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self.prefix
        ):
            found.extend(obj["Key"][len(self.prefix) :] for obj in page.get("Contents", []))
        return sorted(name for name in found if name.endswith(".parquet") and "/" not in name)

    def put(self, name: str, table: pa.Table) -> int:
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        payload = buffer.getvalue()
        if self._s3 is None:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(payload)
        else:
            self._s3.put_object(Bucket=self.bucket, Key=self.prefix + name, Body=payload)
        return len(payload)

    def delete(self, name: str) -> None:
        if self._s3 is None:
            (self.root / name).unlink()
        else:
            self._s3.delete_object(Bucket=self.bucket, Key=self.prefix + name)

    def as_of(self) -> datetime:
        """The lake's clock: its newest cohort's ``inference_date``."""
        dates = {name[:10] for name in self.names()}
        if not dates:
            raise SystemExit(f"{self.location} holds no seeded {TABLE} table - run `seed` first")
        return datetime.strptime(max(dates), "%Y-%m-%d")

    def params(self) -> Params:
        """The parameters the table was seeded with, from a file's footer."""
        names = [n for n in self.names() if n.startswith(f"{FIRST_COHORT:%Y-%m-%d}-")]
        if not names:
            raise SystemExit(f"{self.location} holds no seeded {TABLE} table - run `seed` first")
        if self._s3 is None:
            metadata = pq.read_schema(self.root / names[0]).metadata
        else:
            body = self._s3.get_object(Bucket=self.bucket, Key=self.prefix + names[0])["Body"]
            metadata = pq.read_schema(io.BytesIO(body.read())).metadata
        return Params.from_json(metadata[META_KEY])


# -- commands -------------------------------------------------------------------


def _write_cohort(lake: Lake, cohort: Cohort, params: Params, state: str) -> list[str]:
    written = []
    total = 0
    for chunk, table in enumerate(cohort_tables(cohort, params, state)):
        name = file_name(cohort.when, state, chunk)
        total += lake.put(name, table)
        written.append(name)
    rows = cohort.columns["is_active"].size
    print(f"wrote {cohort.when:%Y-%m-%d} ({state}): {rows} rows, {len(written)} file(s), {total}B")
    return written


def seed(lake: Lake, params: Params) -> None:
    lake.ensure_buckets()
    written: set[str] = set()
    rows = active_rows = positives = 0
    for cohort in simulate(params.customers):
        known = outcome_known(cohort.when)
        state = "matured" if known else "open"
        written.update(_write_cohort(lake, cohort, params, state))
        rows += cohort.columns["is_active"].size
        if known:
            active_rows += int(cohort.columns["is_active"].sum())
            positives += int(cohort.churned.sum())
    for stale in sorted(set(lake.names()) - written):
        lake.delete(stale)
        print(f"removed stale {stale}")
    width = len(schema(params.noise_columns))
    print(
        f"seeded {lake.location}: {rows} rows x {width} columns; "
        f"churn rate among labelled active rows {positives / max(active_rows, 1):.1%}"
    )


def _rewrite(lake: Lake, cohort: Cohort, params: Params, state: str) -> None:
    before = [n for n in lake.names() if n.startswith(f"{cohort.when:%Y-%m-%d}-")]
    written = _write_cohort(lake, cohort, params, state)
    # Put first, then delete: a reader never sees the cohort missing.
    for name in sorted(set(before) - set(written)):
        lake.delete(name)


def rewrite_newest(lake: Lake, state: str) -> None:
    params = lake.params()
    as_of = lake.as_of()
    *_, newest = simulate(params.customers, until=as_of)
    _rewrite(lake, newest, params, state)


def advance(lake: Lake, weeks: int, *, simulate_future: bool, today: datetime) -> None:
    """Move the lake's clock ``weeks`` Mondays forward, one week at a time.

    Each week the newest cohort's outcome week closes, so it is rewritten with
    its labels, and the next Monday's cohort lands open. A cohort dated after
    ``today`` would describe a population that does not exist yet, so that
    needs ``simulate_future`` - the showcase is honest about which weeks have
    happened.
    """
    params = lake.params()
    as_of = lake.as_of()
    target = as_of + weeks * CADENCE
    if target > today and not simulate_future:
        raise SystemExit(
            f"advancing {weeks} week(s) moves the lake to {target:%Y-%m-%d}, after today "
            f"({today:%Y-%m-%d}): that cohort has not happened yet. Pass --simulate-future "
            "(make advance-week FUTURE=1) to simulate it anyway."
        )
    for cohort in simulate(params.customers, until=target):
        if as_of <= cohort.when < target:
            _rewrite(lake, cohort, params, "matured")
        elif cohort.when == target:
            _rewrite(lake, cohort, params, "open")
    future = " (simulated: after today)" if target > today else ""
    print(f"advanced {lake.location} to {target:%Y-%m-%d}{future}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lake", default=DEFAULT_LAKE, help=f"s3:// prefix or local dir (default {DEFAULT_LAKE})"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    seed_cmd = sub.add_parser("seed", help="write the whole table (newest cohort open)")
    seed_cmd.add_argument("--scale", choices=sorted(SCALES), default="default")
    seed_cmd.add_argument("--customers", type=int, help="override the scale's customer count")
    seed_cmd.add_argument("--noise-columns", type=int, help="override the scale's noise columns")
    seed_cmd.add_argument("--rows-per-file", type=int, default=ROWS_PER_FILE)
    for name, text in (
        ("land-outcomes", "rewrite the newest cohort with its realized labels"),
        ("inject-drift", "rewrite the newest cohort with numeric features x3"),
        ("reset", "rewrite the newest cohort open again"),
    ):
        sub.add_parser(name, help=text)
    advance_cmd = sub.add_parser(
        "advance", help="a week passes: label the newest cohort, land the next one open"
    )
    advance_cmd.add_argument("--weeks", type=int, default=1)
    advance_cmd.add_argument(
        "--simulate-future",
        action="store_true",
        help="allow cohorts dated after today (a simulated future)",
    )
    advance_cmd.add_argument(
        "--today", type=datetime.fromisoformat, help="the real date (default: today, UTC)"
    )
    args = parser.parse_args(argv)

    lake = Lake(args.lake)
    if args.command == "seed":
        customers, noise = SCALES[args.scale]
        seed(
            lake,
            Params(
                customers=args.customers or customers,
                noise_columns=noise if args.noise_columns is None else args.noise_columns,
                rows_per_file=args.rows_per_file,
            ),
        )
    elif args.command == "advance":
        if args.weeks < 1:
            parser.error("--weeks must be at least 1")
        today = args.today or datetime.now(UTC).replace(tzinfo=None)
        advance(lake, args.weeks, simulate_future=args.simulate_future, today=today)
    else:
        state = {"land-outcomes": "matured", "inject-drift": "drifted", "reset": "open"}
        rewrite_newest(lake, state[args.command])
    return 0


if __name__ == "__main__":
    sys.exit(main())
