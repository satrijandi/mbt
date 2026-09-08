"""Declarative per-feature treatment, applied in the job (ADR-27).

Three blocks under ``features:`` act at three different seams, and two of
them are implemented right here so that every adapter gets them without
knowing they exist:

* ``categorical`` **retypes** a column. Core casts declared categoricals to
  string, which is exactly what every adapter's dtype inference already keys
  off (native categoricals in the tree adapters, ``asfactor`` in H2O,
  ``StringIndexer`` in Spark), so an int-coded category stops training as an
  ordinal magnitude. Level policies that need no fitted state - pinned
  ``levels``, ``null_as_level``, the ``max_levels`` guard - are string
  rewrites and live here too; ``min_frequency`` needs the train-fitted level
  map and therefore lives in ``mbt_adapter_base.encoding`` instead.
* ``transforms`` **rewrites values**: ``cap`` -> ``log`` -> ``percentile``,
  in that fixed order, on the same table at train and at score time. Every
  step is stateless, so the two sides agree without a side-car artifact.
* ``monotonic`` constrains the *model*, not the data, so it is not applied
  here at all - the adapters read it off the spec and hand it to the booster.

Nulls survive every step as nulls, so the tree adapters keep using their
own missing branch. numpy loads lazily (ADR-14); cap and log stay in
pyarrow so an integer column is not silently widened to float.
"""

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from mbt.exceptions import ConfigError
from mbt_adapter_base.specs import (
    MISSING_LEVEL,
    OTHER_LEVEL,
    CategoricalPolicy,
    FeatureSelection,
    FeatureTransform,
)

if TYPE_CHECKING:
    import numpy as np

#: Arrow dtype prefixes a declared categorical may be cast from. Floating
#: point is excluded deliberately: "1.0" is a poor category key and almost
#: always means the wrong column was declared.
_CATEGORICAL_SOURCE_PREFIXES = ("int", "uint", "bool", "string", "large_string")

#: Arrow dtype prefixes a numeric transform may be applied to.
_NUMERIC_PREFIXES = ("int", "uint", "float", "double", "decimal")

#: The split whose cardinality the ``max_levels`` guard measures. It exists to
#: catch an identifier declared categorical, which is a modelling mistake
#: visible at train time; a scoring batch that widens later is the shift
#: monitor's job, not a reason to fail a production run.
_GUARDED_SPLIT = "train"


def apply_treatment(
    table: pa.Table,
    selection: FeatureSelection,
    features: list[str],
    split: str,
    *,
    resource: str | None = None,
) -> pa.Table:
    """Retype declared categoricals, then rewrite transformed numerics.

    ``features`` is the post-selection feature list; the target and declared
    slice columns are in ``table`` but are never treated. Returns the table
    with the same columns in the same order.
    """
    _check_treated_columns_exist(selection, features, resource)
    table = _apply_categoricals(table, selection, features, split, resource)
    return _apply_transforms(table, selection, resource)


def _check_treated_columns_exist(
    selection: FeatureSelection, features: list[str], resource: str | None
) -> None:
    """A treated column the model never sees is always a mistake - typically a
    typo, or a column dropped by an ``include`` glob that did not match."""
    available = set(features)
    missing = [name for name in selection.treated_columns if name not in available]
    if missing:
        raise ConfigError(
            "feature treatment names column(s) the model does not consume: "
            f"{', '.join(sorted(missing))}",
            resource=resource,
            hint=(
                "'categorical', 'transforms' and 'monotonic' treat the columns that "
                f"survive features.include/exclude, which here are: {', '.join(features)}"
            ),
        )


# -- categoricals ---------------------------------------------------------------


def _apply_categoricals(
    table: pa.Table,
    selection: FeatureSelection,
    features: list[str],
    split: str,
    resource: str | None,
) -> pa.Table:
    if not selection.declares_categorical:
        return table  # dtype inference, unchanged (the pre-ADR-27 behaviour)
    policies = selection.categorical_policies
    _reject_undeclared_strings(table, policies, features, resource)
    for name, policy in policies.items():
        index = table.column_names.index(name)
        column = _as_levels(table.column(index), name, policy, resource)
        if split == _GUARDED_SPLIT:
            _check_cardinality(column, name, policy, resource)
        table = table.set_column(index, name, column)
    return table


def _reject_undeclared_strings(
    table: pa.Table,
    policies: dict[str, CategoricalPolicy],
    features: list[str],
    resource: str | None,
) -> None:
    """Once ``features.categorical`` is present it is the whole truth, so a
    string feature the DS did not declare is an error rather than a guess.

    This is what makes ``categorical: []`` a meaningful assertion instead of a
    synonym for omitting the key.
    """
    undeclared = [
        name
        for name in features
        if name not in policies and str(table.schema.field(name).type) in ("string", "large_string")
    ]
    if undeclared:
        raise ConfigError(
            f"string feature(s) not declared in features.categorical: {', '.join(undeclared)}",
            resource=resource,
            hint=(
                "features.categorical is declared, so it is authoritative - add these "
                "columns to it, or drop them via features.exclude"
            ),
        )


def _as_levels(
    column: pa.ChunkedArray, name: str, policy: CategoricalPolicy, resource: str | None
) -> pa.ChunkedArray:
    """Cast to string, then apply the stateless level policies.

    ``levels`` runs before ``null_as_level`` on purpose: a null is untouched by
    a value rewrite, so ``__missing__`` is never itself pooled into
    ``__other__`` for being an undeclared level.
    """
    dtype = str(column.type)
    if not dtype.startswith(_CATEGORICAL_SOURCE_PREFIXES):
        raise ConfigError(
            f"feature {name!r} is declared categorical but has type {dtype}",
            resource=resource,
            hint=(
                "declare integer-, boolean-, or string-typed columns categorical; a "
                "floating-point or temporal column needs bucketing upstream or in a "
                "hooks.py transform_features first"
            ),
        )
    column = column.cast(pa.string())
    allowed = policy.level_strings
    if allowed is not None:
        # is_in answers False for a null whichever way skip_nulls is set, so
        # the mask is restored afterwards: without that, a null would pool
        # into __other__ for "not being one of the levels" before
        # null_as_level ever saw it.
        pooled = pc.if_else(
            pc.is_in(column, value_set=pa.array(allowed, pa.string())),
            column,
            pa.scalar(OTHER_LEVEL, pa.string()),
        )
        column = pc.if_else(pc.is_valid(column), pooled, pa.scalar(None, pa.string()))
    if policy.null_as_level:
        column = pc.fill_null(column, pa.scalar(MISSING_LEVEL, pa.string()))
    if not isinstance(column, pa.ChunkedArray):  # pragma: no cover - pc returns either
        column = pa.chunked_array([column])
    return column


def _check_cardinality(
    column: pa.ChunkedArray, name: str, policy: CategoricalPolicy, resource: str | None
) -> None:
    if policy.max_levels is None:
        return
    distinct = pc.count_distinct(column).as_py() or 0
    if distinct > policy.max_levels:
        raise ConfigError(
            f"categorical {name!r} has {distinct} distinct levels in the train "
            f"split, above max_levels ({policy.max_levels})",
            resource=resource,
            hint=(
                "a column with this many levels is usually an identifier rather than "
                "a category - drop it via features.exclude, or pool the tail with "
                "min_frequency"
            ),
        )


# -- numeric transforms ---------------------------------------------------------


def _apply_transforms(
    table: pa.Table, selection: FeatureSelection, resource: str | None
) -> pa.Table:
    for name, transform in selection.transforms.items():
        if not transform.rewrites_values:
            continue  # a monotonic-only entry constrains the model, not the data
        index = table.column_names.index(name)
        column = table.column(index)
        dtype = str(column.type)
        if not dtype.startswith(_NUMERIC_PREFIXES):
            raise ConfigError(
                f"feature {name!r} has a numeric transform but has type {dtype}",
                resource=resource,
                hint=(
                    "cap/log/percentile need a magnitude; declare the column under "
                    "features.categorical instead, or exclude it"
                ),
            )
        table = table.set_column(index, name, _transform_column(column, name, transform, resource))
    return table


def _transform_column(
    column: pa.ChunkedArray, name: str, transform: FeatureTransform, resource: str | None
) -> pa.ChunkedArray:
    """The fixed order: cap, then log, then percentile."""
    if transform.cap is not None:
        if transform.cap.min is not None:
            column = pc.max_element_wise(
                column, _bound(transform.cap.min, column.type), skip_nulls=False
            )
        if transform.cap.max is not None:
            column = pc.min_element_wise(
                column, _bound(transform.cap.max, column.type), skip_nulls=False
            )
    if transform.log:
        column = _log1p(column, name, resource)
    if transform.percentile is not None:
        column = _percentile_rank(column)
    if not isinstance(column, pa.ChunkedArray):  # pragma: no cover - pc returns either
        column = pa.chunked_array([column])
    return column


def _bound(value: float, dtype: pa.DataType) -> pa.Scalar:
    """The cap as a scalar of the column's own type where that is lossless.

    Capping an integer column at an integral bound must not widen it to double:
    the type survives into the staged parquet the JVM adapters read and into
    the monitoring baseline's numeric/categorical split.
    """
    if pa.types.is_integer(dtype) and float(value).is_integer():
        return pa.scalar(int(value), dtype)
    return pa.scalar(value)


def _log1p(column: pa.ChunkedArray, name: str, resource: str | None) -> pa.ChunkedArray:
    """``log1p``, refusing values below -1 rather than emitting silent NaNs."""
    minimum = pc.min(column).as_py()
    if minimum is not None and minimum <= -1:
        raise ConfigError(
            f"feature {name!r} has log: true but its minimum value is {minimum}",
            resource=resource,
            hint=(
                "log1p(x) is undefined at or below x = -1; give the column a floor "
                "first, e.g. cap: {min: 0}"
            ),
        )
    return pc.log1p(column.cast(pa.float64()))


def _percentile_rank(column: pa.ChunkedArray) -> pa.ChunkedArray:
    """Average-tie rank within THIS split or batch, scaled into (0, 1].

    Ranking against the batch itself is the whole point (ADR-27): a uniform
    shift of the population - every subscriber one month older - moves every
    value together, so the ranks are unchanged and the feature is stationary
    by construction.

    Ties take the average rank (the scipy/pandas default), which matters
    because a plateau cap deliberately creates one large tie group: ``min``
    ranks would push the whole capped tail to the bottom of its own group.
    NaN is folded into the missing mask alongside null - neither has a
    position in an ordering - and comes back out as null.
    """
    import numpy as np

    values = column.cast(pa.float64()).to_numpy(zero_copy_only=False)
    missing = np.isnan(values)
    present = ~missing
    count = int(present.sum())
    ranks = np.full(values.shape, np.nan, dtype=np.float64)
    if count:
        ranks[present] = _average_ranks(values[present]) / count
    return pa.chunked_array([pa.array(ranks, type=pa.float64(), mask=missing)])


def _average_ranks(values: "np.ndarray") -> "np.ndarray":
    """1-based ranks with ties averaged, vectorised (no per-tie-group loop:
    a batch of a million distinct values would make that quadratic-ish)."""
    import numpy as np

    count = values.shape[0]
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts_group = np.empty(count, dtype=bool)
    starts_group[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts_group[1:])
    group_of = np.cumsum(starts_group) - 1
    starts = np.flatnonzero(starts_group)
    sizes = np.diff(np.append(starts, count))
    # The mean of the 1-based ranks start+1 .. start+size.
    group_rank = starts + (sizes + 1) / 2.0
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = group_rank[group_of]
    return ranks
