"""mbt-evidently: a real Evidently drift run, its payload parsing, and the
plugin's import hygiene (ADR-14, ADR-30)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from mbt_evidently.engine import EvidentlyReportingEngine, parse_snapshot

from mbt_adapter_base import CONTRACT_VERSION


def _frame(rng: np.random.Generator, n: int, *, shift: float, levels: list[str]) -> pa.Table:
    return pa.table(
        {
            "tenure": rng.normal(shift, 1.0, n),
            "plan": rng.choice(levels, n).tolist(),
            "prediction": rng.random(n),
        }
    )


def test_a_shifted_period_drifts_and_gets_its_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    rng = np.random.default_rng(3)
    reference = _frame(rng, 300, shift=0.0, levels=["basic", "pro"])
    current = _frame(rng, 200, shift=2.0, levels=["basic", "pro", "enterprise"])
    out = tmp_path / "2026-05.html"

    found = EvidentlyReportingEngine({}).drift_report(
        reference, current, out, title="churn: 2026-05 against test"
    )

    assert os.environ["DO_NOT_TRACK"] == "1"  # set before evidently loads
    by_column = {c.column: c for c in found.columns}
    assert set(by_column) == {"tenure", "plan", "prediction"}
    assert by_column["tenure"].drifted and by_column["plan"].drifted
    assert not by_column["prediction"].drifted  # same distribution on both sides
    # small samples get statistical tests, which drift below their threshold
    assert by_column["tenure"].method == "K-S p_value"
    assert by_column["tenure"].score < by_column["tenure"].threshold
    assert found.drifted_share == pytest.approx(2 / 3)
    page = out.read_text(encoding="utf-8")
    assert page.lstrip().startswith("<!DOCTYPE html>")
    assert "churn: 2026-05 against test" in page


def _entry(kind: str, value: Any, **config: Any) -> dict[str, Any]:
    return {"config": {"type": f"evidently:metric_v2:{kind}", **config}, "value": value}


def test_distances_drift_at_their_threshold_and_p_values_below_it() -> None:
    found = parse_snapshot(
        {
            "metrics": [
                _entry("DriftedColumnsCount", {"count": 1.0, "share": 0.25}),
                _entry(
                    "ValueDrift",
                    0.1,
                    column="a",
                    method="Wasserstein distance (normed)",
                    threshold=0.1,
                ),
                _entry(
                    "ValueDrift", 0.09, column="b", method="Jensen-Shannon distance", threshold=0.1
                ),
                _entry("ValueDrift", 0.05, column="c", method="K-S p_value", threshold=0.05),
                _entry(
                    "ValueDrift", 0.049, column="d", method="chi-square p_value", threshold=0.05
                ),
                # anything else the preset adds is not a column verdict
                _entry("ValueDrift", {"unexpected": True}, column="e", method="x", threshold=1),
                {"metric_name": "SomethingNew()", "value": 3},
            ]
        }
    )
    assert found.drifted_share == 0.25
    assert {c.column: c.drifted for c in found.columns} == {
        "a": True,
        "b": False,
        "c": False,
        "d": True,
    }


def test_a_payload_without_the_drift_count_is_a_version_problem() -> None:
    with pytest.raises(ValueError, match="payload shape changed"):
        parse_snapshot(
            {"metrics": [_entry("ValueDrift", 0.2, column="a", method="m", threshold=0.1)]}
        )


def test_the_plugin_names_the_role_and_stays_out_of_the_env_digest() -> None:
    from mbt_evidently.plugin import PLUGIN

    assert PLUGIN.name == "evidently"
    assert PLUGIN.contract_version == CONTRACT_VERSION
    assert PLUGIN.reporting is EvidentlyReportingEngine
    assert PLUGIN.fingerprint_packages == []
    assert PLUGIN.training is None and PLUGIN.tuning is None


def test_importing_the_plugin_does_not_load_evidently() -> None:
    probe = (
        "import sys, mbt_evidently.plugin as p; "
        "p.PLUGIN.reporting({}); "
        "print('evidently' in sys.modules, 'pandas' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert proc.stdout.split() == ["False", "False"]
