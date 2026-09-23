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
from pathlib import Path
from typing import Any

import pyarrow as pa

from mbt.events import get_bus
from mbt.events.models import LogMessage
from mbt.exceptions import ConfigError
from mbt.execute.feature_treatment import apply_treatment
from mbt.quality.hooks import ModelHooks
from mbt_adapter_base import (
    OUT_OF_TIME_SPLIT,
    DatasetHandle,
    DatasetLocator,
    DatasetProfile,
    HookContext,
    ModelSpec,
)


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


class TrainingSplitView:
    """The part of a materialization the training path is allowed to see.

    Two rules, both applied once here so no consumer can forget them:

    - the after-test split does not exist (ADR-30). Training, the validation
      and calibration carves, path-adapter staging and hooks all enumerate
      ``splits()``; none of them may touch rows reserved for judging the model
      after its test window - the same guarantee ADR-8 gives the test split
      against tuning;
    - ``evaluation.protocol.test_window`` narrows the test split to its
      resolved window. The spec always documented this; before this view the
      window was resolved and then silently ignored.

    The training report builds the same view with ``hide_out_of_time=False``:
    it is the one reader the after-test split exists for, and it must see the
    test split exactly as the metrics did.
    """

    def __init__(
        self,
        base: DatasetHandle,
        *,
        time_column: str | None,
        test_window: tuple[str, str] | None = None,
        hide_out_of_time: bool = True,
    ) -> None:
        self._base = base
        self._time_column = time_column
        self._test_window = test_window
        self._hidden = {OUT_OF_TIME_SPLIT} if hide_out_of_time else set()

    @property
    def snapshot_id(self) -> str:
        return self._base.snapshot_id

    @property
    def time_column(self) -> str | None:
        return self._time_column

    @property
    def label_column(self) -> str:
        return str(getattr(self._base, "label_column", ""))

    def splits(self) -> set[str]:
        return {s for s in self._base.splits() if s not in self._hidden}

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        if split in self._hidden:
            raise ConfigError(
                f"split {OUT_OF_TIME_SPLIT!r} is reserved for the training report "
                "and is never read by the training path (ADR-30)"
            )
        if split != "test" or self._test_window is None:
            return self._base.read(split, columns)
        if not self._time_column:
            raise ConfigError(
                "evaluation.protocol.test_window needs the split time column, but the "
                "materialization records none",
                hint="test_window applies to temporal splits only",
            )
        from mbt_adapter_base.reporting import window_mask

        table = self._base.read(split)
        start, end = self._test_window
        table = table.filter(window_mask(table.column(self._time_column), start, end))
        return table if columns is None else table.select(columns)

    def profile(self) -> DatasetProfile:
        profile = self._base.profile()
        n_rows = {s: n for s, n in profile.n_rows.items() if s not in self._hidden}
        return profile.model_copy(update={"n_rows": n_rows})

    def locator(self) -> DatasetLocator:
        return self._base.locator()


class SplitRouter:
    """Serves each split from the handle that already holds it (ADR-30).

    The report scores train and test through the handle the model was fit and
    evaluated with - already staged to parquet for a path adapter - and only
    the after-test split through a fresh one. ``split_path`` follows the same
    route, staging a split whose handle has no file of its own on demand.
    """

    def __init__(self, routes: dict[str, DatasetHandle]) -> None:
        self._routes = dict(routes)

    @property
    def snapshot_id(self) -> str:
        return next(iter(self._routes.values())).snapshot_id

    def splits(self) -> set[str]:
        return set(self._routes)

    def read(self, split: str, columns: list[str] | None = None) -> pa.Table:
        return self._routes[split].read(split, columns)

    def split_path(self, split: str) -> Path:
        from mbt_adapter_base.training_helpers import staged_split_path

        return staged_split_path(self._routes[split], split, prefix="mbt-report-split-")

    def profile(self) -> DatasetProfile:
        return next(iter(self._routes.values())).profile()

    def locator(self) -> DatasetLocator:
        return next(iter(self._routes.values())).locator()


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
        pinned_features: list[str] | None = None,
    ) -> None:
        self._base = base
        self._spec = spec
        self._hooks = hooks
        self._hook_ctx_factory = hook_ctx_factory
        self._time_column = time_column
        #: False for scoring inputs: unlabeled by design (ADR-20).
        self._require_target = require_target
        #: The champion's recorded feature columns (ADR-28's
        #: ``resolved.feature_columns``), when scoring against one. Set, it is
        #: authoritative over the globs: the champion was fit on exactly these,
        #: in exactly this order. ``None`` at train time and for a champion
        #: registered before mbt exported an inference config, which restores
        #: the glob-only behaviour exactly.
        #: Normalised: an EMPTY recorded list is a MISSING pin, not a pin of
        #: zero features (C-2). The write side and the read side used to
        #: disagree about what empty meant - ``_apply_pin`` treats any non-None
        #: pin as authoritative, so an empty list would project every batch
        #: onto no columns at all.
        self._pinned_features = pinned_features or None
        self._pin_warned = False
        self._cache: dict[str, pa.Table] = {}
        self._feature_columns: list[str] | None = None

    @property
    def snapshot_id(self) -> str:
        return self._base.snapshot_id

    @property
    def base(self) -> Any:
        """The untransformed view underneath (it carries the time column).

        Public because ``job.py`` reached into ``_base`` eight times, once just
        to format a log line (C-2). A caller that needs the raw rows - the
        carves, the backtest folds - is not doing anything illegitimate.
        """
        return self._base

    def feature_columns(self, split: str | None = None) -> list[str]:
        """The feature columns this view exposes, resolving them if needed.

        This was a public ATTRIBUTE initialised to ``None`` and populated as a
        side effect of the first ``read()``, so seven readers relied on someone
        having read first and every one fell back to ``or []`` - and ``job.py``
        carried the line ``runtime.transformed.read("train")  # resolves the
        feature columns``, an obligation held in a comment (C-2).

        Transformation is cached, so asking costs at most one read of one split.
        Returns ``[]`` only when the view genuinely has no features.
        """
        if self._feature_columns is None:
            available = self.splits()
            chosen = split or ("train" if "train" in available else min(available, default=""))
            if not chosen:
                return []
            self._transformed(chosen)
        return list(self._feature_columns or [])

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
        if self._pinned_features is not None:
            features = self._apply_pin(features, table.column_names, split)
        if self._feature_columns is None:
            self._feature_columns = features
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

    def _apply_pin(self, features: list[str], columns: list[str], split: str) -> list[str]:
        """Project onto the champion's recorded feature columns (ADR-28/ADR-29).

        Missing is fatal and extra is not, deliberately. A batch that lost a
        trained feature can only produce garbage, and used to surface as a raw
        ``KeyError`` out of whichever adapter indexed it first. A batch that
        gained a column is the normal state of a panel whose upstream shipped
        the next feature before the retrain landed, so it is dropped with one
        warning. Ordering matters for the positional consumers (Spark's
        ``VectorAssembler``, H2O's column list); the arrow adapters index by
        name and do not care.
        """
        pinned = list(self._pinned_features or [])
        missing = [c for c in pinned if c not in columns]
        if missing:
            raise ConfigError(
                f"scoring input is missing feature(s) the champion was trained on: "
                f"{', '.join(missing)}",
                resource=self._spec.name,
                hint=(
                    f"the champion was fit on {len(pinned)} feature(s); add the "
                    "column(s) upstream, or retrain and promote against the "
                    "current input schema"
                ),
            )
        extra = [c for c in features if c not in pinned]
        if extra and not self._pin_warned:
            self._pin_warned = True
            shown = ", ".join(extra[:5]) + (", ..." if len(extra) > 5 else "")
            get_bus().emit(
                LogMessage(
                    level="warn",
                    unique_id=self._spec.name,
                    message=(
                        f"scoring input has {len(extra)} column(s) the champion was "
                        f"not trained on ({shown}); ignoring them for split {split!r}. "
                        "Retrain and promote to take them as features"
                    ),
                )
            )
        return pinned

    def profile(self) -> DatasetProfile:
        return self._base.profile()

    def locator(self) -> DatasetLocator:
        return self._base.locator()
