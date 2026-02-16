"""Structured logging for MBT framework.

Emits JSON-formatted logs for:
- Step execution events
- Pipeline events
- Errors and warnings
"""

import json
import sys
from datetime import datetime
from typing import Any


class StructuredLogger:
    """Structured logger that emits JSON logs."""

    def __init__(self, enabled: bool = True, output=None):
        """Initialize structured logger.

        Args:
            enabled: Whether to enable logging
            output: Output stream (default: sys.stderr)
        """
        self.enabled = enabled
        self.output = output or sys.stderr

    def _log(self, level: str, event: str, **kwargs):
        """Emit a structured log entry.

        Args:
            level: Log level (INFO, WARNING, ERROR)
            event: Event name
            **kwargs: Additional fields
        """
        if not self.enabled:
            return

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
            **kwargs,
        }

        print(json.dumps(log_entry), file=self.output)

    def log_pipeline_start(self, pipeline: str, run_id: str, target: str):
        """Log pipeline start event."""
        self._log(
            "INFO",
            "pipeline_started",
            pipeline=pipeline,
            run_id=run_id,
            target=target,
        )

    def log_pipeline_complete(self, pipeline: str, run_id: str, duration: float, status: str):
        """Log pipeline completion event."""
        self._log(
            "INFO",
            "pipeline_completed",
            pipeline=pipeline,
            run_id=run_id,
            duration_seconds=duration,
            status=status,
        )

    def log_step_start(self, pipeline: str, step: str, run_id: str):
        """Log step start event."""
        self._log(
            "INFO",
            "step_started",
            pipeline=pipeline,
            step=step,
            run_id=run_id,
        )

    def log_step_complete(self, pipeline: str, step: str, run_id: str, duration: float):
        """Log step completion event."""
        self._log(
            "INFO",
            "step_completed",
            pipeline=pipeline,
            step=step,
            run_id=run_id,
            duration_seconds=duration,
        )

    def log_step_failure(self, pipeline: str, step: str, run_id: str, error: str, duration: float):
        """Log step failure event."""
        self._log(
            "ERROR",
            "step_failed",
            pipeline=pipeline,
            step=step,
            run_id=run_id,
            error=error,
            duration_seconds=duration,
        )

    def log_artifact_stored(self, artifact_name: str, uri: str, run_id: str, step: str):
        """Log artifact storage event."""
        self._log(
            "INFO",
            "artifact_stored",
            artifact_name=artifact_name,
            uri=uri,
            run_id=run_id,
            step=step,
        )

    def log_artifact_loaded(self, artifact_name: str, uri: str, run_id: str, step: str):
        """Log artifact load event."""
        self._log(
            "INFO",
            "artifact_loaded",
            artifact_name=artifact_name,
            uri=uri,
            run_id=run_id,
            step=step,
        )

    def log_warning(self, message: str, **kwargs):
        """Log warning."""
        self._log("WARNING", "warning", message=message, **kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error."""
        self._log("ERROR", "error", message=message, **kwargs)


# Global logger instance
_logger = StructuredLogger(enabled=False)  # Disabled by default


def get_logger() -> StructuredLogger:
    """Get global structured logger instance."""
    return _logger


def enable_structured_logging(output=None):
    """Enable structured logging.

    Args:
        output: Output stream (default: sys.stderr)
    """
    global _logger
    _logger = StructuredLogger(enabled=True, output=output)


def disable_structured_logging():
    """Disable structured logging."""
    global _logger
    _logger.enabled = False
