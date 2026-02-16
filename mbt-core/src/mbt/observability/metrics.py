"""Metrics emission for MBT framework.

Emits metrics to monitoring systems (StatsD, Prometheus, etc.):
- Step execution duration
- Step success/failure counts
- Pipeline execution duration
- Artifact sizes
"""

from typing import Any


class MetricsEmitter:
    """Metrics emitter for monitoring."""

    def __init__(self, enabled: bool = True, backend: str = "statsd", config: dict | None = None):
        """Initialize metrics emitter.

        Args:
            enabled: Whether to enable metrics
            backend: Metrics backend (statsd, prometheus, cloudwatch)
            config: Backend-specific configuration
        """
        self.enabled = enabled
        self.backend = backend
        self.config = config or {}
        self._client = None

        if self.enabled:
            self._init_client()

    def _init_client(self):
        """Initialize metrics client based on backend."""
        # For Phase 6: Stub implementation
        # In production: would initialize actual client (StatsD, Prometheus, etc.)
        pass

    def emit_step_duration(self, pipeline: str, step: str, duration: float):
        """Emit step execution duration metric.

        Args:
            pipeline: Pipeline name
            step: Step name
            duration: Duration in seconds
        """
        if not self.enabled:
            return

        metric_name = f"mbt.pipelines.{pipeline}.{step}.duration_seconds"
        self._emit_gauge(metric_name, duration, tags={"pipeline": pipeline, "step": step})

    def emit_step_status(self, pipeline: str, step: str, status: str):
        """Emit step status metric.

        Args:
            pipeline: Pipeline name
            step: Step name
            status: Status (success, failed)
        """
        if not self.enabled:
            return

        metric_name = f"mbt.pipelines.{pipeline}.{step}.status"
        value = 1 if status == "success" else 0
        self._emit_gauge(metric_name, value, tags={"pipeline": pipeline, "step": step, "status": status})

    def emit_pipeline_duration(self, pipeline: str, duration: float):
        """Emit pipeline execution duration metric.

        Args:
            pipeline: Pipeline name
            duration: Duration in seconds
        """
        if not self.enabled:
            return

        metric_name = f"mbt.pipelines.{pipeline}.duration_seconds"
        self._emit_gauge(metric_name, duration, tags={"pipeline": pipeline})

    def emit_pipeline_status(self, pipeline: str, status: str):
        """Emit pipeline status metric.

        Args:
            pipeline: Pipeline name
            status: Status (success, failed)
        """
        if not self.enabled:
            return

        metric_name = f"mbt.pipelines.{pipeline}.status"
        value = 1 if status == "success" else 0
        self._emit_gauge(metric_name, value, tags={"pipeline": pipeline, "status": status})

    def emit_artifact_size(self, pipeline: str, step: str, artifact: str, size_bytes: int):
        """Emit artifact size metric.

        Args:
            pipeline: Pipeline name
            step: Step name
            artifact: Artifact name
            size_bytes: Size in bytes
        """
        if not self.enabled:
            return

        metric_name = f"mbt.artifacts.{pipeline}.{step}.{artifact}.size_bytes"
        self._emit_gauge(metric_name, size_bytes, tags={"pipeline": pipeline, "step": step, "artifact": artifact})

    def _emit_gauge(self, metric_name: str, value: float, tags: dict | None = None):
        """Emit a gauge metric.

        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        # For Phase 6: Stub implementation
        # In production: would send to actual backend
        # Example for StatsD: self._client.gauge(metric_name, value, tags=tags)
        pass

    def _emit_counter(self, metric_name: str, value: int = 1, tags: dict | None = None):
        """Emit a counter metric.

        Args:
            metric_name: Metric name
            value: Increment value
            tags: Optional tags
        """
        # For Phase 6: Stub implementation
        pass


# Global metrics emitter instance
_emitter = MetricsEmitter(enabled=False)  # Disabled by default


def get_emitter() -> MetricsEmitter:
    """Get global metrics emitter instance."""
    return _emitter


def enable_metrics(backend: str = "statsd", config: dict | None = None):
    """Enable metrics emission.

    Args:
        backend: Metrics backend (statsd, prometheus, cloudwatch)
        config: Backend-specific configuration
    """
    global _emitter
    _emitter = MetricsEmitter(enabled=True, backend=backend, config=config)


def disable_metrics():
    """Disable metrics emission."""
    global _emitter
    _emitter.enabled = False
