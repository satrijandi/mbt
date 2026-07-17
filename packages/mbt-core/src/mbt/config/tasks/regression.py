"""The regression task schema (TSD §5.6).

Regression drops binary classification's two-class label check: the target is a
continuous numeric column, validated by dtype rather than class count, and no
``scale_pos_weight`` applies.
"""

from mbt.contracts import (
    DatasetProfile,
    ModelSpec,
    TaskType,
    ValidationIssue,
)
from mbt_adapter_base.metrics import REGRESSION_METRIC_BASES

#: Arrow dtype-string prefixes that count as a numeric regression target.
_NUMERIC_ARROW_PREFIXES = ("int", "uint", "float", "double", "decimal", "halffloat")


class RegressionSchema:
    """Validates model specs and dataset profiles for regression."""

    task = TaskType.REGRESSION

    @property
    def allowed_metrics(self) -> set[str]:
        return set(REGRESSION_METRIC_BASES)

    def validate_spec(self, spec: ModelSpec) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for i, slice_column in enumerate(spec.evaluation.slices):
            if slice_column == spec.target:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        resource=spec.name,
                        field_path=f"/evaluation/slices/{i}",
                        message="slicing by the target column is meaningless",
                        hint="slice by a feature column such as region or plan_type",
                    )
                )
        return issues

    def validate_dataset(self, spec: ModelSpec, profile: DatasetProfile) -> list[ValidationIssue]:
        """The target must be a numeric column (no class-count check)."""
        issues: list[ValidationIssue] = []
        dtype = profile.columns.get(profile.label_column, "")
        if not dtype.startswith(_NUMERIC_ARROW_PREFIXES):
            issues.append(
                ValidationIssue(
                    severity="error",
                    resource=spec.name,
                    field_path="/target",
                    message=(
                        f"regression requires a numeric target; label "
                        f"'{profile.label_column}' has dtype {dtype or 'unknown'!r}"
                    ),
                    hint="cast the label to a numeric column in the source or a hooks.py transform",
                )
            )
        return issues
