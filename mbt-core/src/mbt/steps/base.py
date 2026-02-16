"""Base step class - all pipeline steps inherit from this."""

from abc import ABC, abstractmethod
from typing import Any


class Step(ABC):
    """Base class for all pipeline steps.

    Steps are the unit of execution in MBT. Each step:
    - Receives inputs (artifacts from previous steps)
    - Receives context (configuration, run_id, etc.)
    - Produces outputs (artifacts for downstream steps)
    """

    @abstractmethod
    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Execute the step.

        Args:
            inputs: Dictionary mapping input_name -> artifact
            context: RunContext with config, run_id, execution_date, etc.

        Returns:
            Dictionary mapping output_name -> artifact
        """
        ...
