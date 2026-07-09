"""Categorical feature derivation and encoding for tabular tree adapters.

Shared so XGBoost and LightGBM derive features, category levels, and codes
identically - champion/challenger comparisons stay apples to apples. String
columns train as native categoricals; levels are the sorted unique values of
the train split (deterministic); encoding maps values to float codes with
NaN for missing values and levels unseen at train time (the frameworks'
missing-value branch). numpy loads lazily (ADR-14).
"""

from typing import TYPE_CHECKING

import pyarrow as pa

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


def train_categories(table: pa.Table, categorical: list[str]) -> dict[str, list[str]]:
    """Sorted unique non-null levels per categorical column (train split).

    Sorted so the value-to-code mapping is deterministic across runs and
    machines; the mapping is persisted with the model artifact.
    """
    return {
        name: sorted({str(v) for v in table.column(name).to_pylist() if v is not None})
        for name in categorical
    }


def categorical_codes(table: pa.Table, name: str, levels: list[str]) -> "np.ndarray":
    """Float codes for one column: the level's index, NaN when missing or
    unseen at train time."""
    import numpy as np

    index = {level: float(code) for code, level in enumerate(levels)}
    return np.asarray(
        [
            np.nan if v is None else index.get(str(v), np.nan)
            for v in table.column(name).to_pylist()
        ],
        dtype=np.float64,
    )
