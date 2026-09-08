"""Categorical feature derivation and encoding for tabular tree adapters.

Shared so XGBoost and LightGBM derive features, category levels, and codes
identically - champion/challenger comparisons stay apples to apples. String
columns train as native categoricals; levels are the sorted unique values of
the train split (deterministic); encoding maps values to float codes with
NaN for missing values and levels unseen at train time (the frameworks'
missing-value branch). numpy loads lazily (ADR-14).

Core retypes DS-declared categoricals to string before an adapter ever sees
the table (ADR-27), so everything here still keys off dtype. The one level
policy that cannot live in core is ``min_frequency``: pooling a rare tail
needs the train-fitted level map, and that map is *already* persisted with
the model artifact and reused at predict time, so pooling belongs here where
it round-trips for free rather than behind a new side-car artifact. Once
``__other__`` is a level, a value unseen at train time codes to it instead of
to NaN - the whole point of asking for a pooled bucket.
"""

from typing import TYPE_CHECKING

import pyarrow as pa

from mbt_adapter_base.specs import OTHER_LEVEL, CategoricalPolicy

if TYPE_CHECKING:
    import numpy as np

NUMERIC_PREFIXES = ("int", "uint", "float", "double", "decimal", "bool")
CATEGORICAL_TYPES = ("string", "large_string")


def split_feature_columns(
    table: pa.Table, *, target: str, slices: list[str], adapter: str
) -> tuple[list[str], list[str]]:
    """(feature columns in table order, the categorical subset).

    Features are every column except the target and declared slice columns.
    Numeric columns pass through; string columns are categorical; anything
    else (timestamps, nested types, ...) raises the actionable error.
    """
    features = [n for n in table.column_names if n != target and n not in slices]
    categorical: list[str] = []
    bad: list[str] = []
    for name in features:
        dtype = str(table.schema.field(name).type)
        if dtype.startswith(NUMERIC_PREFIXES):
            continue
        if dtype in CATEGORICAL_TYPES:
            categorical.append(name)
        else:
            bad.append(f"{name} ({dtype})")
    if bad:
        raise ValueError(
            f"unsupported feature column type(s) for {adapter}: {', '.join(bad)}. "
            "Numeric and string (categorical) columns train natively; exclude "
            "others under features.exclude or encode them in a hooks.py "
            "transform_features."
        )
    return features, categorical


def train_categories(
    table: pa.Table,
    categorical: list[str],
    policies: dict[str, CategoricalPolicy] | None = None,
) -> dict[str, list[str]]:
    """Sorted unique non-null levels per categorical column (train split).

    Sorted so the value-to-code mapping is deterministic across runs and
    machines; the mapping is persisted with the model artifact.

    A column whose declared policy sets ``min_frequency`` has its rare levels
    replaced by a single trailing ``__other__`` (ADR-27), so a long tail of
    one-off values cannot each buy a split.
    """
    policies = policies or {}
    levels: dict[str, list[str]] = {}
    for name in categorical:
        values = [str(v) for v in table.column(name).to_pylist() if v is not None]
        minimum = policies[name].min_frequency if name in policies else None
        levels[name] = _pool_rare(values, minimum)
    return levels


def _pool_rare(values: list[str], min_frequency: float | None) -> list[str]:
    """Sorted distinct levels, with those below ``min_frequency`` of the rows
    collapsed into a trailing ``__other__``.

    ``__other__`` sorts last rather than alphabetically so the code of a real
    level does not shift when the pooled bucket appears or disappears between
    retrains - a level's code is a persisted part of the artifact.
    """
    distinct = sorted(set(values))
    if min_frequency is None or not values:
        return distinct
    total = len(values)
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    kept = [level for level in distinct if counts[level] / total >= min_frequency]
    if len(kept) == len(distinct):
        return distinct
    return [*kept, OTHER_LEVEL]


def categorical_codes(table: pa.Table, name: str, levels: list[str]) -> "np.ndarray":
    """Float codes for one column: the level's index, NaN when missing.

    A value unseen at train time codes to ``__other__`` when that level
    exists - a pooled bucket that did not catch unseen values would leave
    them in the missing branch, which is the opposite of what asking for one
    means - and to NaN otherwise, which is the default and unchanged.
    """
    import numpy as np

    index = {level: float(code) for code, level in enumerate(levels)}
    unseen = index.get(OTHER_LEVEL, np.nan)
    return np.asarray(
        [
            np.nan if v is None else index.get(str(v), unseen)
            for v in table.column(name).to_pylist()
        ],
        dtype=np.float64,
    )
