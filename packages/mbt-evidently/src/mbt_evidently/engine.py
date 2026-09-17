"""Evidently's data-drift report as an mbt report engine (ADR-30).

The training report's stability section is mbt's own PSI and KS, and those
numbers decide every gate. This engine adds what a data scientist opens
Evidently for: its per-column tests, chosen by column type and sample size,
and its interactive HTML. Nothing here gates; a drifted column is shown.

``evidently`` is imported inside ``drift_report`` only (ADR-14), and the
job process opts out of Evidently's usage telemetry before it loads.
"""

import html
import os
import warnings
from pathlib import Path
from typing import Any

import pyarrow as pa

from mbt_adapter_base import DriftColumn, DriftReport


class EvidentlyReportingEngine:
    """``evaluation.report.stability.engine: evidently``."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def drift_report(
        self, reference: pa.Table, current: pa.Table, out_html: Path, *, title: str
    ) -> DriftReport:
        # Only Evidently's UI service and collector send usage events today;
        # the variable is its documented opt-out, set before the import in
        # case a release moves that into the library path.
        os.environ.setdefault("DO_NOT_TRACK", "1")
        with warnings.catch_warnings():
            # Import-time deprecations from its web stack, and statistics that
            # divide by empty categories on small cells: noise in a job log.
            warnings.simplefilter("ignore")
            from evidently import Report
            from evidently.presets import DataDriftPreset

            snapshot = Report([DataDriftPreset()]).run(
                current_data=current.to_pandas(), reference_data=reference.to_pandas()
            )
            page = snapshot.get_html_str(as_iframe=False)
        # The page has no <title>; a reader with a tab per month needs one.
        heading = f"<head><title>{html.escape(title)}</title>"
        out_html.write_text(page.replace("<head>", heading, 1), encoding="utf-8")
        return parse_snapshot(snapshot.dict())


def parse_snapshot(payload: dict[str, Any]) -> DriftReport:
    """The drift verdicts in an Evidently 0.7 snapshot.

    Each column is a ``ValueDrift`` entry, ``{"config": {"type":
    "evidently:metric_v2:ValueDrift", "column": "a", "method": "K-S p_value",
    "threshold": 0.05}, "value": 1.1e-08}`` (captured from 0.7.23). A
    p-value drifts below its threshold, a distance at or above it.
    """
    columns: list[DriftColumn] = []
    share: float | None = None
    for entry in payload.get("metrics", []):
        config = entry.get("config") or {}
        kind = str(config.get("type", ""))
        value = entry.get("value")
        if kind.endswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value["share"])
        elif kind.endswith("ValueDrift") and isinstance(value, int | float):
            method = str(config.get("method", ""))
            threshold = float(config.get("threshold", 0.0))
            score = float(value)
            drifted = score < threshold if "p_value" in method else score >= threshold
            columns.append(
                DriftColumn(
                    column=str(config.get("column", "")),
                    method=method,
                    score=score,
                    threshold=threshold,
                    drifted=drifted,
                )
            )
    if share is None:
        raise ValueError(
            "the evidently snapshot has no DriftedColumnsCount metric; "
            "its payload shape changed - check the mbt-evidently version pin"
        )
    return DriftReport(drifted_share=share, columns=columns)
