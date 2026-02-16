"""Runtime context - passed to steps during execution."""

from typing import Any, Optional
from datetime import datetime


class RunContext:
    """Context object passed to step.run() method.

    Contains configuration, runtime variables, and helper methods.
    """

    def __init__(
        self,
        config: dict[str, Any],
        run_id: str,
        execution_date: Optional[datetime] = None,
        variables: Optional[dict[str, Any]] = None,
    ):
        self.config = config
        self.run_id = run_id
        self.execution_date = execution_date or datetime.utcnow()
        self.variables = variables or {}

    def get_config(self, *keys: str, default: Any = None) -> Any:
        """Get nested config value.

        Example:
            context.get_config('training', 'model_training', 'framework')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_var(self, key: str, default: Any = None) -> Any:
        """Get runtime variable (from --vars flag)."""
        return self.variables.get(key, default)
