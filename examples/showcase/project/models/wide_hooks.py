"""Shared hooks for the wide models (SHOW-20): DS-declared categorical codes.

contract_code is a numeric-coded categorical (int8, 0 = month-to-month ...
3 = two-year) whose churn effect is non-monotone, so treating the code as a
number costs real signal. mbt has no `categorical:` spec field on purpose -
adapters infer categoricals from dtype - and this hook is the sanctioned
seam for numeric-coded ones: casting to string here makes every adapter's
native handling pick the column up (LightGBM native categoricals, H2O
asfactor, SparkML StringIndexer).

The cast runs per split BEFORE include/exclude filtering and again at
scoring time, and this file's bytes hash into both wide models' config
hashes (editing it retrains them; mbt.hooks_hash parity guards a champion
trained with a different version).

scripts/select_features.py imports CATEGORICAL_CODES from this file and
applies the same cast before the selection funnel, so one declared list
governs selection, training, and scoring.
"""

import pyarrow as pa
import pyarrow.compute as pc

#: Numeric-coded categorical columns, declared by the DS.
CATEGORICAL_CODES: list[str] = ["contract_code"]


def transform_features(table: pa.Table, ctx) -> pa.Table:
    for name in CATEGORICAL_CODES:
        if name in table.column_names:
            idx = table.column_names.index(name)
            table = table.set_column(idx, name, pc.cast(table.column(idx), pa.string()))
    return table
