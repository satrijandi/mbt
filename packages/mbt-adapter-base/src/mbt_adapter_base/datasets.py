"""In-memory DatasetHandle: tuning carves, compliance fixtures, tests."""

import pyarrow as pa

from mbt_adapter_base.interchange import DatasetLocator, DatasetProfile


class InMemoryDatasetHandle:
    """A DatasetHandle over plain Arrow tables (not locatable/serializable)."""

    def __init__(
        self,
        tables: dict[str, pa.Table],
        snapshot_id: str = "sha256:inmemory",
        label_column: str = "",
        time_column: str | None = None,
    ) -> None:
        self._tables = dict(tables)
        self._snapshot_id = snapshot_id
        self._label_column = label_column
        self._time_column = time_column
        self._profile: DatasetProfile | None = None

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def label_column(self) -> str:
        return self._label_column

    @property
    def time_column(self) -> str | None:
        return self._time_column

    def splits(self) -> set[str]:
        return set(self._tables)

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        table = self._tables[split]
        if columns is None:
            return table
        return table.select(columns)

    def with_split(self, split: str, table: pa.Table) -> "InMemoryDatasetHandle":
        tables = dict(self._tables)
        tables[split] = table
        return InMemoryDatasetHandle(
            tables,
            snapshot_id=self._snapshot_id,
            label_column=self._label_column,
            time_column=self._time_column,
        )

    def profile(self) -> DatasetProfile:
        if self._profile is None:
            self._profile = self._compute_profile()
        return self._profile

    def _compute_profile(self) -> DatasetProfile:
        n_rows = {split: table.num_rows for split, table in self._tables.items()}
        any_table = next(iter(self._tables.values()))
        columns = {field.name: str(field.type) for field in any_table.schema}
        label_balance: dict[str, float] | None = None
        label = self._label_column
        train = self._tables.get("train")
        if label and train is not None and label in train.column_names and train.num_rows:
            values = [str(v) for v in train.column(label).to_pylist()]
            total = len(values)
            label_balance = {cls: values.count(cls) / total for cls in sorted(set(values))}
        time_range: tuple[str, str] | None = None
        if self._time_column and self._time_column in columns:
            lows, highs = [], []
            for table in self._tables.values():
                if table.num_rows == 0:
                    continue
                column = table.column(self._time_column).to_pylist()
                lows.append(min(column))
                highs.append(max(column))
            if lows:
                time_range = (str(min(lows)), str(max(highs)))
        return DatasetProfile(
            n_rows=n_rows,
            columns=columns,
            label_column=label,
            label_balance=label_balance,
            time_range=time_range,
        )

    def locator(self) -> DatasetLocator:
        raise NotImplementedError(
            "in-memory datasets are not locatable; they never cross a process boundary"
        )
