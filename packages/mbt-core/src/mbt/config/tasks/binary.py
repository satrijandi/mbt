"""The binary_classification task schema (TSD §5.6)."""

from mbt_adapter_base.metrics import BINARY_METRIC_BASES, is_builtin_binary_metric

from mbt.contracts import (
    DatasetProfile,
    ModelSpec,
    TaskType,
    ValidationIssue,
)

_BINARY_LABEL_VALUES = {"0", "1", "0.0", "1.0", "false", "true"}


class BinaryClassificationSchema:
    """Validates model specs and dataset profiles for binary classification."""

    task = TaskType.BINARY_CLASSIFICATION

    @property
    def allowed_metrics(self) -> set[str]:
        return set(BINARY_METRIC_BASES)

    def is_allowed_metric(self, name: str) -> bool:
        """Builtin check including parameterized sugar (recall_at_precision_0.9)."""
        return is_builtin_binary_metric(name)

    def validate_spec(self, spec: ModelSpec) -> list[ValidationIssue]:
        """Parse-time protocol sanity. Metric-name resolution happens in the
        parser's metric resolution step, which also knows metrics.yml and hooks."""
        issues: list[ValidationIssue] = []
        for i, slice_column in enumerate(spec.evaluation.slices):
            if slice_column == spec.target:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        resource=spec.name,
                        field_path=f"/evaluation/slices/{i}",
                        message="slicing by the target column is meaningless",
                        hint="slice by a feature column such as plan_type or region",
                    )
                )
        return issues

    def validate_dataset(
        self, spec: ModelSpec, profile: DatasetProfile
    ) -> list[ValidationIssue]:
        """Run-time validation once the dataset profile exists."""
        issues: list[ValidationIssue] = []
        balance = profile.label_balance or {}
        classes = set(balance)
        if len(classes) != 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    resource=spec.name,
                    field_path="/target",
                    message=(
                        f"binary classification requires a binary label; "
                        f"'{profile.label_column}' has classes {sorted(classes)}"
                    ),
                    hint="check the dataset label definition and filters",
                )
            )
        elif not {c.lower() for c in classes} <= _BINARY_LABEL_VALUES:
            issues.append(
                ValidationIssue(
                    severity="error",
                    resource=spec.name,
                    field_path="/target",
                    message=(
                        f"label '{profile.label_column}' must be encoded as 0/1 or bool, "
                        f"got classes {sorted(classes)}"
                    ),
                    hint="encode the label in the dataset source or a hooks.py transform",
                )
            )
        else:
            minority = min(balance.values())
            if minority < 0.001:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        resource=spec.name,
                        field_path="/target",
                        message=f"extreme class imbalance: minority class is {minority:.4%}",
                        hint="consider scale_pos_weight: '{{ auto }}' or resampling upstream",
                    )
                )
        return issues
