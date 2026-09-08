"""Dataset handle wrappers applied inside the training job (TSD §10.5).

``transform_features`` applies per split after read; ``features.include/
exclude`` then applies to the post-hook column set. The table an adapter
finally reads contains exactly: selected features + target + declared slice
columns; the split time column is always dropped from features (TSD §5.6).

Declarative feature treatment (ADR-27) runs last, on the projected table:
``features.categorical`` retypes, ``features.transforms`` rewrites values.
Doing it here rather than in each adapter is what makes it uniform - every
construction site of this class (train, the validation and calibration
carves, each walk-forward fold, and scoring) goes through one code path, and
``_materialize_for_path_adapter`` stages this table for the JVM adapters, so
Spark and H2O see treated data without knowing the feature exists.
"""

from collections.abc import Callable
from fnmatch import fnmatchcase

import pyarrow as pa

from mbt.contracts import (
    DatasetHandle,
    DatasetLocator,
    DatasetProfile,
    HookContext,
    ModelSpec,
)
from mbt.exceptions import ConfigError
from mbt.execute.feature_treatment import apply_treatment
from mbt.quality.hooks import ModelHooks


def select_feature_columns(
    columns: list[str],
    spec: ModelSpec,
    time_column: str | None,
) -> list[str]:
    """Apply include/exclude globs; target and time column never count as features."""
    never = {spec.target}
    if time_column:
        never.add(time_column)
    included = [
        c
        for c in columns
        if c not in never and any(fnmatchcase(c, pattern) for pattern in spec.features.include)
    ]
    features = [
        c for c in included if not any(fnmatchcase(c, pattern) for pattern in spec.features.exclude)
    ]
    if not features:
        raise ConfigError(
            f"feature selection left no columns for model {spec.name!r}",
            hint=(
                f"include={spec.features.include} exclude={spec.features.exclude} "
                f"over columns {columns}"
            ),
        )
    return features


class TransformedDatasetHandle:
    """Applies hooks and feature selection lazily, per split."""

    def __init__(
        self,
        base: DatasetHandle,
        spec: ModelSpec,
        hooks: ModelHooks | None,
        hook_ctx_factory: "Callable[[str], HookContext]",
        time_column: str | None,
        *,
        require_target: bool = True,
    ) -> None:
        self._base = base
        self._spec = spec
        self._hooks = hooks
        self._hook_ctx_factory = hook_ctx_factory
        self._time_column = time_column
        #: False for scoring inputs: unlabeled by design (ADR-20).
        self._require_target = require_target
        self._cache: dict[str, pa.Table] = {}
        self.feature_columns: list[str] | None = None

    @property
    def snapshot_id(self) -> str:
        return self._base.snapshot_id

    def splits(self) -> set[str]:
        return self._base.splits()

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        table = self._transformed(split)
        if columns is None:
            return table
        return table.select(columns)

    def _transformed(self, split: str) -> pa.Table:
        if split in self._cache:
            return self._cache[split]
        table = self._base.read(split)
        if self._hooks is not None and self._hooks.has_transform:
            ctx: HookContext = self._hook_ctx_factory(split)
            table = self._hooks.transform_features(table, ctx)
        features = select_feature_columns(table.column_names, self._spec, self._time_column)
        if self.feature_columns is None:
            self.feature_columns = features
        keep = list(features)
        for extra in (self._spec.target, *self._spec.evaluation.slices):
            if extra in table.column_names and extra not in keep:
                keep.append(extra)
        if self._require_target and self._spec.target not in table.column_names:
            raise ConfigError(
                f"target column {self._spec.target!r} missing after hooks for split {split!r}",
                hint="transform_features must preserve the target column",
            )
        table = table.select(keep)
        # Declared slice columns ride along for evaluation but are not features
        # (the adapters drop them), so they are neither treatable nor subject to
        # the authoritative-categorical rule.
        slices = set(self._spec.evaluation.slices)
        table = apply_treatment(
            table,
            self._spec.features,
            [c for c in features if c not in slices],
            split,
            resource=self._spec.name,
        )
        self._cache[split] = table
        return table

    def profile(self) -> DatasetProfile:
        return self._base.profile()

    def locator(self) -> DatasetLocator:
        return self._base.locator()
