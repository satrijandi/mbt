"""Step executor - single-step execution for distributed (pod-per-step) pipelines.

Used by `mbt step execute` to run one step at a time, with artifacts
persisted to remote storage (S3) between steps. Each pod reads its
inputs from the artifact registry, executes its step, and writes outputs back.
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime
from importlib import import_module
from typing import Any

from mbt.core.manifest import Manifest
from mbt.core.context import RunContext
from mbt.core.registry import PluginRegistry


class StepExecutor:
    """Executes a single pipeline step with remote artifact persistence.

    Unlike the full Runner which executes all steps in-process, StepExecutor
    handles one step at a time. The artifact registry is persisted to storage
    between steps so that separate processes/pods can share artifacts.
    """

    def __init__(self, manifest: Manifest, project_root: Path, run_id: str | None = None):
        self.manifest = manifest
        self.project_root = Path(project_root)
        self.profile_config = manifest.profile_config

        # Resolve storage plugin from profile config
        registry = PluginRegistry()
        storage_config = self.profile_config.get("storage", {})
        storage_type = storage_config.get("type", "local")
        self.storage = registry.get("mbt.storage", storage_type)
        self.storage.configure(storage_config.get("config", {}))

        self.run_id = run_id or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Load artifact registry from storage (persisted between pods)
        self.artifact_registry = self._load_artifact_registry()

    def execute_step(self, step_name: str) -> dict[str, Any]:
        """Execute a single step.

        Args:
            step_name: Name of the step to execute

        Returns:
            Dict with status, run_id, and step results
        """
        if step_name not in self.manifest.steps:
            raise ValueError(
                f"Step '{step_name}' not found in manifest. "
                f"Available steps: {list(self.manifest.steps.keys())}"
            )

        step_def = self.manifest.steps[step_name]
        print(f"Executing step: {step_name}")
        print(f"  Run ID: {self.run_id}")

        start_time = time.time()

        try:
            # Load input artifacts
            inputs = self._load_inputs(step_def.inputs)

            # Create context
            context = RunContext(
                config=step_def.config,
                run_id=self.run_id,
                profile_config=self.profile_config,
            )

            # Load and instantiate step class
            step_instance = self._load_step_class(step_def.plugin)

            # Execute step
            outputs = step_instance.run(inputs, context)

            # Store output artifacts
            self._store_outputs(step_name, step_def.outputs, outputs)

            # Persist artifact registry for next step
            self._save_artifact_registry()

            duration = time.time() - start_time
            print(f"  Step completed in {duration:.2f}s")

            return {
                "status": "success",
                "run_id": self.run_id,
                "step": step_name,
                "duration_seconds": duration,
            }

        except Exception as e:
            duration = time.time() - start_time
            print(f"  Step failed after {duration:.2f}s: {e}")
            raise

    def _load_inputs(self, input_names: list[str]) -> dict[str, Any]:
        """Load and deserialize input artifacts from storage."""
        inputs = {}
        for name in input_names:
            if name not in self.artifact_registry:
                raise RuntimeError(
                    f"Input artifact '{name}' not found in artifact registry. "
                    f"Available: {list(self.artifact_registry.keys())}"
                )

            uri = self.artifact_registry[name]
            data = self.storage.get(uri)
            inputs[name] = pickle.loads(data)

        return inputs

    def _store_outputs(self, step_name: str, output_names: list[str], outputs: dict[str, Any]):
        """Serialize and store output artifacts."""
        for name in output_names:
            if name not in outputs:
                raise RuntimeError(f"Step {step_name} did not produce expected output: {name}")

            data = pickle.dumps(outputs[name])
            uri = self.storage.put(
                artifact_name=name,
                data=data,
                run_id=self.run_id,
                step_name=step_name,
            )
            self.artifact_registry[name] = uri

    def _load_step_class(self, plugin_path: str):
        """Dynamically import and instantiate step class."""
        module_path, class_name = plugin_path.split(":")
        module = import_module(module_path)
        step_class = getattr(module, class_name)
        return step_class()

    def _load_artifact_registry(self) -> dict[str, str]:
        """Load artifact registry from storage."""
        registry_uri = self._registry_uri()
        if self.storage.exists(registry_uri):
            data = self.storage.get(registry_uri)
            return json.loads(data)
        return {}

    def _save_artifact_registry(self):
        """Persist artifact registry to storage."""
        data = json.dumps(self.artifact_registry).encode()
        self.storage.put(
            artifact_name=".artifact_registry.json",
            data=data,
            run_id=self.run_id,
            step_name="_meta",
        )

    def _registry_uri(self) -> str:
        """Get the URI for the artifact registry in storage."""
        storage_config = self.profile_config.get("storage", {})
        storage_type = storage_config.get("type", "local")

        if storage_type == "s3":
            bucket = storage_config.get("config", {}).get("bucket", "mbt-pipeline-artifacts")
            return f"s3://{bucket}/{self.run_id}/_meta/.artifact_registry.json"
        else:
            from pathlib import Path
            base_path = storage_config.get("config", {}).get("base_path", "./local_artifacts")
            full_path = Path(base_path) / self.run_id / "_meta" / ".artifact_registry.json"
            return f"file://{full_path.absolute()}"
