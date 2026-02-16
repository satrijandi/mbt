"""DAG assertion framework for testing pipeline compilation.

Supports assertions:
- step_exists: Check if a step exists in the DAG
- step_absent: Check if a step does NOT exist in the DAG
- step_order: Check if one step comes before another
- step_count: Check if number of steps is within range
- resource_limit: Check if step has resource limits
"""

from typing import Any
from pydantic import BaseModel, Field


class TestResult(BaseModel):
    """Result of a single assertion test."""

    assertion_type: str
    passed: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class DagAssertion(BaseModel):
    """Base class for DAG assertions."""

    type: str = Field(..., description="Assertion type")


class StepExistsAssertion(DagAssertion):
    """Assert that a step exists in the DAG."""

    step: str = Field(..., description="Step name to check")


class StepAbsentAssertion(DagAssertion):
    """Assert that a step does NOT exist in the DAG."""

    step: str = Field(..., description="Step name that should not exist")


class StepOrderAssertion(DagAssertion):
    """Assert that one step comes before another."""

    before: str = Field(..., description="Step that should come first")
    after: str = Field(..., description="Step that should come after")


class StepCountAssertion(DagAssertion):
    """Assert that number of steps is within range."""

    min: int | None = Field(None, description="Minimum number of steps")
    max: int | None = Field(None, description="Maximum number of steps")


class ResourceLimitAssertion(DagAssertion):
    """Assert that a step has resource limits."""

    step: str = Field(..., description="Step name to check")
    memory_max: str | None = Field(None, description="Max memory (e.g., '64Gi')")
    cpu_max: str | None = Field(None, description="Max CPU (e.g., '4')")


def run_assertions(manifest: dict, assertions: list[dict]) -> list[TestResult]:
    """Run all assertions against a manifest.

    Args:
        manifest: Compiled manifest dictionary
        assertions: List of assertion dictionaries

    Returns:
        List of TestResult objects
    """
    results = []

    for assertion_dict in assertions:
        assertion_type = assertion_dict.get("type")

        if assertion_type == "step_exists":
            result = _check_step_exists(manifest, assertion_dict)
        elif assertion_type == "step_absent":
            result = _check_step_absent(manifest, assertion_dict)
        elif assertion_type == "step_order":
            result = _check_step_order(manifest, assertion_dict)
        elif assertion_type == "step_count":
            result = _check_step_count(manifest, assertion_dict)
        elif assertion_type == "resource_limit":
            result = _check_resource_limit(manifest, assertion_dict)
        else:
            result = TestResult(
                assertion_type=assertion_type or "unknown",
                passed=False,
                message=f"Unknown assertion type: {assertion_type}",
            )

        results.append(result)

    return results


def _check_step_exists(manifest: dict, assertion: dict) -> TestResult:
    """Check if a step exists in the DAG."""
    step_name = assertion["step"]
    steps = manifest.get("steps", {})

    exists = step_name in steps

    return TestResult(
        assertion_type="step_exists",
        passed=exists,
        message=f"Step '{step_name}' {'exists' if exists else 'does not exist'} in DAG",
        details={"step": step_name, "exists": exists},
    )


def _check_step_absent(manifest: dict, assertion: dict) -> TestResult:
    """Check if a step does NOT exist in the DAG."""
    step_name = assertion["step"]
    steps = manifest.get("steps", {})

    absent = step_name not in steps

    return TestResult(
        assertion_type="step_absent",
        passed=absent,
        message=f"Step '{step_name}' {'is absent' if absent else 'exists (should be absent)'} in DAG",
        details={"step": step_name, "absent": absent},
    )


def _check_step_order(manifest: dict, assertion: dict) -> TestResult:
    """Check if one step comes before another."""
    before_step = assertion["before"]
    after_step = assertion["after"]

    # Get execution order from DAG
    dag = manifest.get("dag", {})
    execution_batches = dag.get("execution_batches", [])

    # Flatten batches to get linear order
    execution_order = []
    for batch in execution_batches:
        execution_order.extend(batch)

    # Check if both steps exist
    if before_step not in execution_order or after_step not in execution_order:
        missing = []
        if before_step not in execution_order:
            missing.append(before_step)
        if after_step not in execution_order:
            missing.append(after_step)

        return TestResult(
            assertion_type="step_order",
            passed=False,
            message=f"Steps missing from DAG: {missing}",
            details={"before": before_step, "after": after_step, "missing": missing},
        )

    # Get positions
    before_pos = execution_order.index(before_step)
    after_pos = execution_order.index(after_step)

    correct_order = before_pos < after_pos

    return TestResult(
        assertion_type="step_order",
        passed=correct_order,
        message=(
            f"Step '{before_step}' comes before '{after_step}'"
            if correct_order
            else f"Step '{before_step}' comes AFTER '{after_step}' (should be before)"
        ),
        details={
            "before": before_step,
            "after": after_step,
            "before_position": before_pos,
            "after_position": after_pos,
        },
    )


def _check_step_count(manifest: dict, assertion: dict) -> TestResult:
    """Check if number of steps is within range."""
    min_steps = assertion.get("min")
    max_steps = assertion.get("max")

    steps = manifest.get("steps", {})
    step_count = len(steps)

    passed = True
    reasons = []

    if min_steps is not None and step_count < min_steps:
        passed = False
        reasons.append(f"has {step_count} steps (minimum {min_steps})")

    if max_steps is not None and step_count > max_steps:
        passed = False
        reasons.append(f"has {step_count} steps (maximum {max_steps})")

    if passed:
        message = f"Step count {step_count} is within range"
        if min_steps is not None and max_steps is not None:
            message += f" [{min_steps}-{max_steps}]"
        elif min_steps is not None:
            message += f" (>= {min_steps})"
        elif max_steps is not None:
            message += f" (<= {max_steps})"
    else:
        message = f"Step count validation failed: {', '.join(reasons)}"

    return TestResult(
        assertion_type="step_count",
        passed=passed,
        message=message,
        details={"step_count": step_count, "min": min_steps, "max": max_steps},
    )


def _check_resource_limit(manifest: dict, assertion: dict) -> TestResult:
    """Check if a step has resource limits."""
    step_name = assertion["step"]
    expected_memory = assertion.get("memory_max")
    expected_cpu = assertion.get("cpu_max")

    steps = manifest.get("steps", {})

    if step_name not in steps:
        return TestResult(
            assertion_type="resource_limit",
            passed=False,
            message=f"Step '{step_name}' not found in DAG",
            details={"step": step_name},
        )

    step = steps[step_name]
    resources = step.get("resources", {})

    passed = True
    reasons = []

    if expected_memory is not None:
        actual_memory = resources.get("memory")
        if actual_memory != expected_memory:
            passed = False
            reasons.append(f"memory: expected '{expected_memory}', got '{actual_memory}'")

    if expected_cpu is not None:
        actual_cpu = resources.get("cpu")
        if actual_cpu != expected_cpu:
            passed = False
            reasons.append(f"cpu: expected '{expected_cpu}', got '{actual_cpu}'")

    if passed:
        message = f"Step '{step_name}' has expected resource limits"
    else:
        message = f"Step '{step_name}' resource limits mismatch: {', '.join(reasons)}"

    return TestResult(
        assertion_type="resource_limit",
        passed=passed,
        message=message,
        details={
            "step": step_name,
            "resources": resources,
            "expected_memory": expected_memory,
            "expected_cpu": expected_cpu,
        },
    )
