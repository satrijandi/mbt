"""Runner - executes pipeline steps with artifact passing via storage.

The runner:
1. Loads manifest
2. For each step in execution order:
   a. Loads input artifacts from storage
   b. Deserializes artifacts into Python objects
   c. Executes step.run(inputs, context)
   d. Serializes output artifacts
   e. Stores outputs via StoragePlugin
3. Generates run_results.json
"""

import time
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from importlib import import_module

from mbt.core.manifest import Manifest
from mbt.core.context import RunContext
from mbt.core.registry import PluginRegistry


class RunResults:
    """Results of a pipeline run."""

    def __init__(self, run_id: str, pipeline_name: str, target: str):
        self.run_id = run_id
        self.pipeline_name = pipeline_name
        self.target = target
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.completed_at: Optional[str] = None
        self.status = "running"
        self.steps: dict[str, dict] = {}

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "target": self.target,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "steps": self.steps,
        }


class Runner:
    """Executes pipeline steps according to manifest."""

    def __init__(self, manifest: Manifest, project_root: Path):
        self.manifest = manifest
        self.project_root = Path(project_root)
        self.profile_config = manifest.profile_config

        # Resolve storage plugin from profile config
        registry = PluginRegistry()
        storage_config = self.profile_config.get("storage", {})
        storage_type = storage_config.get("type", "local")
        self.storage = registry.get("mbt.storage", storage_type)
        self.storage.configure(storage_config.get("config", {}))

        self.run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.artifact_registry: dict[str, str] = {}  # artifact_name -> URI

    def run(self) -> RunResults:
        """Execute all steps in the pipeline.

        Returns:
            RunResults object with execution details
        """
        results = RunResults(
            run_id=self.run_id,
            pipeline_name=self.manifest.metadata.pipeline_name,
            target=self.manifest.metadata.target,
        )

        print(f"\n🚀 Starting pipeline: {self.manifest.metadata.pipeline_name}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Target: {self.manifest.metadata.target}\n")

        # Execute steps in order from execution_batches
        for batch in self.manifest.dag.execution_batches:
            for step_name in batch:
                step_result = self._execute_step(step_name)
                results.steps[step_name] = step_result

        # Check if any steps failed
        failed_steps = [name for name, result in results.steps.items() if result["status"] == "failed"]
        results.completed_at = datetime.utcnow().isoformat() + "Z"

        if failed_steps:
            results.status = "failed"
            print(f"\n❌ Pipeline failed. Failed steps: {', '.join(failed_steps)}\n")
        else:
            results.status = "success"
            print(f"\n✅ Pipeline completed successfully\n")

        # Save run results
        self._save_run_results(results)

        return results

    def _execute_step(self, step_name: str) -> dict:
        """Execute a single step.

        Returns:
            Step result dictionary with status, duration, etc.
        """
        step_def = self.manifest.steps[step_name]
        print(f"▶ Executing step: {step_name}")

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

            duration = time.time() - start_time
            print(f"  ✓ Completed in {duration:.2f}s\n")

            return {
                "status": "success",
                "duration_seconds": duration,
            }

        except Exception as e:
            duration = time.time() - start_time
            print(f"  ✗ Failed after {duration:.2f}s: {str(e)}\n")

            return {
                "status": "failed",
                "duration_seconds": duration,
                "error": str(e),
            }

    def _load_inputs(self, input_names: list[str]) -> dict[str, Any]:
        """Load and deserialize input artifacts."""
        inputs = {}
        for name in input_names:
            if name not in self.artifact_registry:
                raise RuntimeError(f"Input artifact not found: {name}")

            uri = self.artifact_registry[name]
            data = self.storage.get(uri)
            inputs[name] = pickle.loads(data)

        return inputs

    def _store_outputs(self, step_name: str, output_names: list[str], outputs: dict[str, Any]):
        """Serialize and store output artifacts."""
        for name in output_names:
            if name not in outputs:
                raise RuntimeError(f"Step {step_name} did not produce expected output: {name}")

            # Serialize artifact
            data = pickle.dumps(outputs[name])

            # Store via storage plugin
            uri = self.storage.put(
                artifact_name=name,
                data=data,
                run_id=self.run_id,
                step_name=step_name,
            )

            # Register URI for future steps
            self.artifact_registry[name] = uri

    def _load_step_class(self, plugin_path: str):
        """Dynamically import and instantiate step class.

        Args:
            plugin_path: "module.path:ClassName"

        Returns:
            Step instance
        """
        module_path, class_name = plugin_path.split(":")
        module = import_module(module_path)
        step_class = getattr(module, class_name)
        return step_class()

    def _save_run_results(self, results: RunResults):
        """Save run_results.json to target directory."""
        output_dir = self.project_root / "target" / self.manifest.metadata.pipeline_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "run_results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"📊 Run results saved to: {results_path}")
