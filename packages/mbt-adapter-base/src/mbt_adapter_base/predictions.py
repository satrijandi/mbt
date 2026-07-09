"""The shared prediction-store layout any DataAdapter can produce (ADR-21).

A prediction store is a directory of runs::

    <root>/<run_key>/predictions.parquet   # the scored rows
    <root>/<run_key>/predictions.json      # PredictionRunInfo sidecar
    <root>/<run_key>/_SUCCESS              # completeness marker
    <root>/<run_key>/<name>.marker.json    # ledger markers (ground_truth, ...)

``write_run`` is idempotent by ``run_key``: rewriting a key replaces the
whole run directory, including its markers - a fresh run gets a fresh
ledger. The local DuckDB adapter uses this layout directly; warehouse
adapters can reuse it for staged exports or implement the
``PredictionStore`` protocol natively.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from mbt_adapter_base.interchange import PredictionRunInfo
from mbt_adapter_base.materialization import SUCCESS_FILE

PREDICTIONS_FILE = "predictions.parquet"
INFO_FILE = "predictions.json"


class PredictionStoreError(RuntimeError):
    """A prediction run is missing or incomplete."""


class LocalPredictionStore:
    """PredictionStore over a local directory (parquet + JSON sidecars)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _run_dir(self, run_key: str) -> Path:
        return self.root / run_key

    def _marker_path(self, run_key: str, name: str) -> Path:
        return self._run_dir(run_key) / f"{name}.marker.json"

    def write_run(self, table: pa.Table, info: PredictionRunInfo) -> PredictionRunInfo:
        run_dir = self._run_dir(info.run_key)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        pq.write_table(table, run_dir / PREDICTIONS_FILE)
        persisted = info.model_copy(
            update={"uri": f"file://{run_dir.resolve()}", "row_count": table.num_rows}
        )
        (run_dir / INFO_FILE).write_text(persisted.model_dump_json(indent=2))
        (run_dir / SUCCESS_FILE).write_text("")
        return persisted

    def list_runs(self) -> list[PredictionRunInfo]:
        runs: list[PredictionRunInfo] = []
        for info_path in self.root.glob(f"*/{INFO_FILE}"):
            if not (info_path.parent / SUCCESS_FILE).is_file():
                continue  # incomplete write; ignored, a re-run replaces it
            runs.append(PredictionRunInfo.model_validate_json(info_path.read_text()))
        return sorted(runs, key=lambda r: (r.scored_at, r.run_key))

    def read(self, run_key: str, columns: list[str] | None = None) -> pa.Table:
        run_dir = self._run_dir(run_key)
        if not (run_dir / SUCCESS_FILE).is_file():
            raise PredictionStoreError(f"no complete prediction run {run_key!r} under {self.root}")
        return pq.read_table(run_dir / PREDICTIONS_FILE, columns=columns)

    def read_marker(self, run_key: str, name: str) -> dict[str, Any] | None:
        path = self._marker_path(run_key, name)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict)
        return payload

    def write_marker(self, run_key: str, name: str, payload: dict[str, Any]) -> None:
        run_dir = self._run_dir(run_key)
        if not (run_dir / SUCCESS_FILE).is_file():
            raise PredictionStoreError(
                f"cannot mark {run_key!r}: no complete prediction run under {self.root}"
            )
        self._marker_path(run_key, name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
