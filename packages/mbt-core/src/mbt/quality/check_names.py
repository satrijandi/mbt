"""Authoritative built-in dataset-check names (single source of truth).

Kept deliberately import-light - no ``duckdb``/``pyarrow`` - so the parse path
(``parsing/project_parser``) can validate check names without importing the
check *implementations* in ``checks.py``, which pull the data libraries (ADR-14
import hygiene: the parser stays framework-free).

``checks.py`` builds its dispatch table (``_CHECKS``) against these names, and a
unit test pins the two in sync; the parser validates user specs against them.
Adding a built-in check means adding its name here and its function in
``checks.py`` - the sync test fails loudly if only one side changes.
"""

#: Every built-in dataset check a user may declare (TSD §11.1).
BUILTIN_CHECK_NAMES = frozenset(
    {
        "schema",
        "not_null",
        "no_future_columns",
        "label_leakage_scan",
        "class_balance_report",
    }
)

#: Checks valid on a scoring input: it has no label, so label-dependent checks
#: are rejected (ADR-20). A strict subset of :data:`BUILTIN_CHECK_NAMES`.
SCORING_CHECK_NAMES = frozenset({"schema", "not_null", "no_future_columns"})
