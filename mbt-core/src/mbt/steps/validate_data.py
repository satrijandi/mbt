"""Validate data step - performs data quality checks.

Built-in checks:
- null_threshold: Maximum percentage of null values allowed
- value_range: Numeric values must be within range
- expected_columns: Required columns must exist
- unique_key: Column must have unique values
- type_check: Column must match expected data type

Custom checks:
- Import from lib/ directory for project-specific validation
"""

from typing import Any
import pandas as pd
import numpy as np

from mbt.steps.base import Step


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ValidateDataStep(Step):
    """Validate data quality.

    Supports built-in and custom validation checks.
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Validate data quality.

        Returns:
            {"validated_data": MBTFrame} - Same data if validation passes
        """
        data = inputs["raw_data"]
        df = data.to_pandas()

        # Get validation configuration
        validation_config = context.get_config("validation", default={})
        on_failure = validation_config.get("on_failure", "fail")
        checks = validation_config.get("checks", [])

        print(f"  Running {len(checks)} validation checks...")

        failed_checks = []

        # Run each validation check
        for i, check in enumerate(checks):
            check_type = check.get("type")

            try:
                passed, message = self._run_check(df, check, context)

                if passed:
                    print(f"    ✓ Check {i+1}/{len(checks)}: {check_type} - {message}")
                else:
                    print(f"    ✗ Check {i+1}/{len(checks)}: {check_type} - {message}")
                    failed_checks.append((check_type, message))
            except Exception as e:
                message = f"Error running check: {e}"
                print(f"    ✗ Check {i+1}/{len(checks)}: {check_type} - {message}")
                failed_checks.append((check_type, message))

        # Handle failures
        if failed_checks:
            failure_summary = "\n".join([f"  - {check}: {msg}" for check, msg in failed_checks])

            if on_failure == "fail":
                raise ValidationError(
                    f"Data validation failed ({len(failed_checks)} checks):\n{failure_summary}"
                )
            elif on_failure == "warn":
                print(f"  ⚠ Warning: {len(failed_checks)} validation checks failed")
                print(failure_summary)
            # on_failure == "skip_row" would filter bad rows (not implemented yet)

        print(f"  ✓ Validation complete: {len(checks) - len(failed_checks)}/{len(checks)} checks passed")

        return {"validated_data": data}

    def _run_check(self, df: pd.DataFrame, check: dict, context: Any) -> tuple[bool, str]:
        """Run a single validation check.

        Args:
            df: DataFrame to validate
            check: Check configuration
            context: Runtime context

        Returns:
            (passed, message) tuple
        """
        check_type = check.get("type")

        if check_type == "null_threshold":
            return self._check_null_threshold(df, check)
        elif check_type == "value_range":
            return self._check_value_range(df, check)
        elif check_type == "expected_columns":
            return self._check_expected_columns(df, check)
        elif check_type == "unique_key":
            return self._check_unique_key(df, check)
        elif check_type == "type_check":
            return self._check_type(df, check)
        elif check_type == "custom":
            return self._check_custom(df, check, context)
        else:
            return False, f"Unknown check type: {check_type}"

    def _check_null_threshold(self, df: pd.DataFrame, check: dict) -> tuple[bool, str]:
        """Check that null percentage is below threshold.

        Config:
            columns: List of columns to check
            max_null_pct: Maximum percentage (0.0-1.0)
        """
        columns = check.get("columns", [])
        max_null_pct = check.get("max_null_pct", 0.0)

        for col in columns:
            if col not in df.columns:
                return False, f"Column '{col}' not found"

            null_pct = df[col].isnull().mean()
            if null_pct > max_null_pct:
                return False, f"Column '{col}' has {null_pct:.1%} nulls (max {max_null_pct:.1%})"

        return True, f"Null check passed for {len(columns)} columns"

    def _check_value_range(self, df: pd.DataFrame, check: dict) -> tuple[bool, str]:
        """Check that numeric values are within range.

        Config:
            column: Column to check
            min: Minimum value (optional)
            max: Maximum value (optional)
        """
        column = check.get("column")
        min_val = check.get("min")
        max_val = check.get("max")

        if column not in df.columns:
            return False, f"Column '{column}' not found"

        col_data = df[column].dropna()

        if min_val is not None and (col_data < min_val).any():
            actual_min = col_data.min()
            return False, f"Column '{column}' has values below {min_val} (min: {actual_min})"

        if max_val is not None and (col_data > max_val).any():
            actual_max = col_data.max()
            return False, f"Column '{column}' has values above {max_val} (max: {actual_max})"

        return True, f"Values in '{column}' are within range"

    def _check_expected_columns(self, df: pd.DataFrame, check: dict) -> tuple[bool, str]:
        """Check that expected columns exist.

        Config:
            columns: List of required columns
        """
        expected = check.get("columns", [])
        missing = [col for col in expected if col not in df.columns]

        if missing:
            return False, f"Missing columns: {missing}"

        return True, f"All {len(expected)} expected columns present"

    def _check_unique_key(self, df: pd.DataFrame, check: dict) -> tuple[bool, str]:
        """Check that column has unique values.

        Config:
            column: Column to check
        """
        column = check.get("column")

        if column not in df.columns:
            return False, f"Column '{column}' not found"

        duplicates = df[column].duplicated().sum()
        if duplicates > 0:
            return False, f"Column '{column}' has {duplicates} duplicate values"

        return True, f"Column '{column}' has unique values"

    def _check_type(self, df: pd.DataFrame, check: dict) -> tuple[bool, str]:
        """Check that column has expected data type.

        Config:
            column: Column to check
            expected_type: Expected pandas dtype (int64, float64, object, etc.)
        """
        column = check.get("column")
        expected_type = check.get("expected_type")

        if column not in df.columns:
            return False, f"Column '{column}' not found"

        actual_type = str(df[column].dtype)
        if actual_type != expected_type:
            return False, f"Column '{column}' is {actual_type}, expected {expected_type}"

        return True, f"Column '{column}' has correct type ({expected_type})"

    def _check_custom(self, df: pd.DataFrame, check: dict, context: Any) -> tuple[bool, str]:
        """Run custom validation function.

        Config:
            function: Import path (e.g., "lib.validators.check_distribution")

        Custom function signature:
            def check_func(df: pd.DataFrame, context: dict) -> tuple[bool, str]
        """
        function_path = check.get("function")
        if not function_path:
            return False, "Custom check missing 'function' parameter"

        try:
            # Import custom function
            module_path, func_name = function_path.rsplit(".", 1)

            # Import module dynamically
            import importlib
            import sys

            # Add project root to Python path
            project_root = context.get_config("project_root", default=".")
            if project_root not in sys.path:
                sys.path.insert(0, str(project_root))

            module = importlib.import_module(module_path)
            check_func = getattr(module, func_name)

            # Run custom check
            return check_func(df, context)
        except Exception as e:
            return False, f"Custom check failed: {e}"
