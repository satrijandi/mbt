"""Scoring node compilation: identity, snapshots, manifest schema v2 (ADR-20)."""

import json
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from core_helpers import TEST_ANCHOR, write
from test_compile import compile_demo

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import read_manifest
from mbt.exceptions import StateError

SCORING = "scoring.demo.churn_scoring"
MODEL = "model.demo.churn_model"


def _write_batch(project_dir: Path) -> None:
    base = TEST_ANCHOR.replace(tzinfo=None)
    batch = {
        "user_id": list(range(50)),
        "snapshot_date": [base - timedelta(days=1 + i % 5) for i in range(50)],
        "is_active": [True] * 50,
        "tenure_days": [30 + i * 3 for i in range(50)],
        "monthly_usage": [round(i * 4.2, 2) for i in range(50)],
        "plan_type": [("basic", "pro", "enterprise")[i % 3] for i in range(50)],
    }
    out = project_dir / "data" / "scoring_batch"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(batch), out / "part-000.parquet")


def _write_outcomes(project_dir: Path, *, rows: int = 40) -> None:
    outcomes = {
        "user_id": list(range(rows)),
        "churned": [1 if i % 4 == 0 else 0 for i in range(rows)],
    }
    out = project_dir / "data" / "churn_outcomes"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(outcomes), out / "part-000.parquet")


@pytest.fixture()
def scoring_demo(demo_project: Path) -> Path:
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: scoring_batch
                path: data/scoring_batch/*.parquet
              - name: churn_outcomes
                path: data/churn_outcomes/*.parquet
        """,
    )
    write(
        demo_project / "scoring/churn_scoring.yml",
        """
        scoring:
          - name: churn_scoring
            owner: lifecycle-eng@example.com
            model: ref('churn_model')
            tags: [daily]
            input:
              source: source('lakehouse', 'scoring_batch')
              time_column: snapshot_date
              window: "-7d:now"
            monitors:
              prediction_shift:
                threshold: 0.2
            ground_truth:
              label:
                source: source('lakehouse', 'churn_outcomes')
                column: churned
              join_key: user_id
              maturity: "14d"
              metrics: [pr_auc]
            output:
              path: predictions/churn_scores
              columns: [user_id]
        """,
    )
    _write_batch(demo_project)
    _write_outcomes(demo_project)
    return demo_project


def test_scoring_node_shape(scoring_demo: Path, fake_registry: AdapterRegistry) -> None:
    manifest = compile_demo(scoring_demo, fake_registry)
    node = manifest.nodes[SCORING]
    assert node.resource_type == "scoring"
    assert node.snapshot_id and node.snapshot_id.startswith("sha256:")
    assert node.config["input"]["window"] == "-7d:now"  # expression, not resolution
    assert node.resolved["windows"]["score"] == ["2026-06-24T00:00:00Z", "2026-07-01T00:00:00Z"]
    assert node.depends_on == [
        MODEL,
        "source.demo.lakehouse.churn_outcomes",
        "source.demo.lakehouse.scoring_batch",
    ]
    # source() rendered to unique_ids in the compiled config
    assert node.config["input"]["source"] == "source.demo.lakehouse.scoring_batch"
    assert node.config["model"] == MODEL
    assert node.config_hash and node.input_hash
    assert manifest.metadata.manifest_schema_version == 2
    # both scoring sources are pinned on the manifest
    assert manifest.sources["source.demo.lakehouse.scoring_batch"].snapshot_id
    assert manifest.sources["source.demo.lakehouse.churn_outcomes"].snapshot_id
    # the scoring snapshot excludes the ground-truth table: it equals the
    # input source's own snapshot verbatim (single-source combine rule)
    assert node.snapshot_id == manifest.sources["source.demo.lakehouse.scoring_batch"].snapshot_id


def test_ground_truth_refresh_never_flips_scoring_identity(
    scoring_demo: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(scoring_demo, fake_registry)
    _write_outcomes(scoring_demo, rows=45)  # labels matured for more rows
    after = compile_demo(scoring_demo, fake_registry)
    assert (
        before.sources["source.demo.lakehouse.churn_outcomes"].snapshot_id
        != after.sources["source.demo.lakehouse.churn_outcomes"].snapshot_id
    )
    assert before.nodes[SCORING].input_hash == after.nodes[SCORING].input_hash


def test_input_data_change_flips_scoring_only(
    scoring_demo: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(scoring_demo, fake_registry)
    batch_dir = scoring_demo / "data" / "scoring_batch"
    table = pq.read_table(batch_dir / "part-000.parquet")
    pq.write_table(table.slice(0, 40), batch_dir / "part-000.parquet")
    after = compile_demo(scoring_demo, fake_registry)
    assert before.nodes[SCORING].input_hash != after.nodes[SCORING].input_hash
    assert before.nodes[SCORING].config_hash == after.nodes[SCORING].config_hash
    assert before.nodes[MODEL].input_hash == after.nodes[MODEL].input_hash


def test_model_edit_flips_scoring_transitively(
    scoring_demo: Path, fake_registry: AdapterRegistry
) -> None:
    before = compile_demo(scoring_demo, fake_registry)
    model_yml = scoring_demo / "models/churn_model.yml"
    model_yml.write_text(model_yml.read_text().replace("max_depth: 4", "max_depth: 5"))
    after = compile_demo(scoring_demo, fake_registry)
    assert before.nodes[MODEL].input_hash != after.nodes[MODEL].input_hash
    assert before.nodes[SCORING].input_hash != after.nodes[SCORING].input_hash
    assert before.nodes[SCORING].config_hash == after.nodes[SCORING].config_hash


def test_anchor_drift_changes_no_scoring_hashes(
    scoring_demo: Path, fake_registry: AdapterRegistry
) -> None:
    a = compile_demo(scoring_demo, fake_registry)
    b = compile_demo(scoring_demo, fake_registry, anchor=TEST_ANCHOR + timedelta(days=3))
    assert a.nodes[SCORING].input_hash == b.nodes[SCORING].input_hash
    assert a.nodes[SCORING].resolved != b.nodes[SCORING].resolved  # windows moved


def test_manifest_v2_roundtrip_and_v1_readable(
    scoring_demo: Path, fake_registry: AdapterRegistry
) -> None:
    manifest = compile_demo(scoring_demo, fake_registry)
    reread = read_manifest(manifest.to_json(), source="test")
    assert reread.nodes[SCORING].resource_type == "scoring"

    payload = json.loads(manifest.to_json())
    payload["metadata"]["manifest_schema_version"] = 1  # N-1 stays readable
    del payload["nodes"][SCORING]
    read_manifest(json.dumps(payload), source="test")

    payload["metadata"]["manifest_schema_version"] = 3
    with pytest.raises(StateError, match="manifest_schema_version 3"):
        read_manifest(json.dumps(payload), source="test")
