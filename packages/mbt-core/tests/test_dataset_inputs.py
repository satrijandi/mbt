"""Multi-table dataset inputs (features + label + join key) and reproducible
key-based sampling through the local adapter."""

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from test_execution import invoke

from mbt.adapters.registry import AdapterRegistry
from mbt.contracts import DatasetSpec
from mbt.parsing import parse_project

DS = "dataset.demo.churn_joined"
MODEL = "model.demo.churn_model_joined"


def _write_tables(project_dir: Path, n: int = 300) -> None:
    base = TEST_ANCHOR.replace(tzinfo=None) - timedelta(days=200)
    dates = [base + timedelta(days=(i * 199) % 200) for i in range(n)]
    labels = pa.table(
        {
            "customer_id": list(range(n)),
            "snapshot_date": dates,
            "churned": [1 if (i * 31) % 100 < 25 else 0 for i in range(n)],
        }
    )
    usage = pa.table(
        {
            "customer_id": list(range(n)),
            "snapshot_date": dates,
            "monthly_usage": [round((i * 13.7) % 500, 2) for i in range(n)],
            "support_tickets": [i % 7 for i in range(n)],
        }
    )
    # profile features deliberately MISSING for ids >= n-20 (left-join nulls)
    profile = pa.table(
        {
            "customer_id": list(range(n - 20)),
            "snapshot_date": dates[: n - 20],
            "tenure_days": [30 + (i * 7) % 900 for i in range(n - 20)],
        }
    )
    for name, table in (("labels", labels), ("usage", usage), ("profile", profile)):
        out = project_dir / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / "part-000.parquet")


@pytest.fixture()
def joined_project(demo_project: Path) -> Path:
    _write_tables(demo_project)
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: churn_labels
                path: data/labels/*.parquet
              - name: usage_features
                path: data/usage/*.parquet
              - name: profile_features
                path: data/profile/*.parquet
        """,
    )
    write(
        demo_project / "datasets/churn_joined.yml",
        """
        datasets:
          - name: churn_joined
            inputs:
              label: source('lakehouse', 'churn_labels')
              features:
                - source('lakehouse', 'usage_features')
                - source('lakehouse', 'profile_features')
              join_key: [customer_id, snapshot_date]
            label:
              column: churned
            split:
              strategy: temporal
              time_column: snapshot_date
              train: "-180d:-28d"
              test: "-28d:now"
        """,
    )
    write(
        demo_project / "models/churn_model_joined.yml",
        """
        models:
          - name: churn_model_joined
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_joined')
            target: churned
            hyperparameters: {fake_metric_value: 0.7}
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc]
            seed: 3
        """,
    )
    return demo_project


def test_join_builds_and_trains(joined_project: Path, fake_registry: AdapterRegistry) -> None:
    results = invoke(joined_project, fake_registry, select=["churn_model_joined"])
    assert results.exit_code() == 0
    by_id = {r.unique_id: r for r in results.results}
    assert by_id[DS].status == "success"
    assert by_id[MODEL].status == "success"

    # the materialization holds label + feature columns from both tables
    key = next((joined_project / "target/datasets/churn_joined").iterdir())
    train = pq.read_table(key / "train.parquet")
    assert {
        "customer_id",
        "snapshot_date",
        "churned",
        "monthly_usage",
        "support_tickets",
        "tenure_days",
    } <= set(train.column_names)
    # left join: examples with missing profile features keep NULL tenure
    assert train.column("tenure_days").null_count > 0


def test_dataset_depends_on_all_three_sources(
    joined_project: Path, fake_registry: AdapterRegistry
) -> None:
    from core_helpers import TEST_ANCHOR as _  # noqa: F401
    from test_compile import compile_demo

    manifest = compile_demo(joined_project, fake_registry)
    node = manifest.nodes[DS]
    assert set(node.depends_on) == {
        "source.demo.lakehouse.churn_labels",
        "source.demo.lakehouse.usage_features",
        "source.demo.lakehouse.profile_features",
    }
    assert node.snapshot_id and node.snapshot_id.startswith("sha256:")

    # a feature-table change flips the combined snapshot -> dataset identity
    _write_tables(joined_project, n=320)
    changed = compile_demo(joined_project, fake_registry)
    assert changed.nodes[DS].snapshot_id != node.snapshot_id
    assert changed.nodes[DS].input_hash != node.input_hash
    assert changed.nodes[DS].config_hash == node.config_hash


def test_key_based_sampling_is_reproducible_and_monotone(
    joined_project: Path, fake_registry: AdapterRegistry
) -> None:
    def ids_at(fraction: float) -> set[tuple[int, datetime]]:
        import shutil

        shutil.rmtree(joined_project / "target", ignore_errors=True)
        results = invoke(
            joined_project,
            fake_registry,
            select=["churn_model_joined"],
            cli_vars={"sample_fraction": fraction},
        )
        assert results.exit_code() == 0
        key = next((joined_project / "target/datasets/churn_joined").iterdir())
        rows = set()
        for split in ("train", "test"):
            table = pq.read_table(
                key / f"{split}.parquet", columns=["customer_id", "snapshot_date"]
            )
            rows |= set(
                zip(
                    table.column("customer_id").to_pylist(),
                    table.column("snapshot_date").to_pylist(),
                    strict=True,
                )
            )
        return rows

    half_a = ids_at(0.5)
    half_b = ids_at(0.5)
    assert half_a == half_b  # same fraction -> exactly the same rows
    fifth = ids_at(0.2)
    assert fifth <= half_a  # smaller fractions are subsets (threshold hashing)
    assert 0 < len(fifth) < len(half_a) < 300


def test_source_xor_inputs_is_enforced(joined_project: Path) -> None:
    with pytest.raises(Exception, match="exactly one of 'source'"):
        DatasetSpec.model_validate(
            {
                "name": "bad",
                "label": {"column": "y"},
                "split": {"time_column": "ts", "train": "-30d:-7d", "test": "7d"},
            }
        )


def test_bare_table_name_is_a_parse_error(
    joined_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        joined_project / "datasets/bad_inputs.yml",
        """
        datasets:
          - name: bad_inputs
            inputs:
              label: churn_labels
              features: ["source('lakehouse', 'usage_features')"]
              join_key: customer_id
            label:
              column: churned
            split:
              strategy: temporal
              time_column: snapshot_date
              train: "-180d:-28d"
              test: "-28d:now"
        """,
    )
    parsed = parse_project(joined_project, registry=fake_registry, raise_on_error=False)
    errors = [i for i in parsed.report.errors if "source() reference" in i.message]
    assert errors and errors[0].field_path == "/inputs/label"
