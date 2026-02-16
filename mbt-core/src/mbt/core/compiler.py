"""Compiler - transforms pipeline YAML into executable manifest.

Compilation phases (full 5-phase model):
  1. Resolution: base_pipeline + !include (Phase 3)
  2. Schema validation: Pydantic validation (Phase 1)
  3. Plugin validation: framework.validate_config() (Phase 2)
  4. DAG assembly: step registry + topological sort (Phase 1)
  5. Manifest generation: merge profile config (Phase 3)

All phases are now implemented in Phase 3.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

from mbt.config.schema import PipelineConfig
from mbt.config.profiles import ProfilesLoader, ProfileNotFoundError
from mbt.core.dag import DAGBuilder
from mbt.core.manifest import Manifest, ManifestMetadata
from mbt.core.composition import CompositionResolver
from mbt import __version__


class CompilationError(Exception):
    """Raised when pipeline compilation fails."""
    pass


class Compiler:
    """Compiles pipeline YAML into executable manifest."""

    def __init__(self, project_root: Path, runtime_vars: dict[str, str] | None = None):
        self.project_root = Path(project_root)
        self.pipelines_dir = self.project_root / "pipelines"
        self.target_dir = self.project_root / "target"
        self.dag_builder = DAGBuilder()
        self.runtime_vars = runtime_vars or {}
        self.composition_resolver = CompositionResolver(self.pipelines_dir, runtime_vars=self.runtime_vars)
        self.profiles_loader = ProfilesLoader(self.project_root)

    def compile(
        self,
        pipeline_name: str,
        target: str = "dev",
        profile_name: Optional[str] = None
    ) -> Manifest:
        """Compile pipeline to manifest.

        Args:
            pipeline_name: Name of pipeline (without .yaml extension)
            target: Target environment (dev, staging, prod)
            profile_name: Profile name (defaults to pipeline_name or project directory name)

        Returns:
            Compiled Manifest object

        Raises:
            CompilationError: If compilation fails
        """
        try:
            # Load pipeline YAML
            pipeline_path = self.pipelines_dir / f"{pipeline_name}.yaml"
            if not pipeline_path.exists():
                raise CompilationError(f"Pipeline not found: {pipeline_path}")

            # Phase 1: Resolution (base_pipeline + !include)
            yaml_dict = self._resolve_composition(pipeline_path)

            # Phase 2: Schema validation
            pipeline_config = self._validate_schema(yaml_dict)

            # Phase 3: Plugin validation
            self._validate_plugins(pipeline_config)

            # Phase 4: DAG assembly
            steps, dag = self._build_dag(pipeline_config)

            # NEW: Load and resolve profile configuration
            resolved_profile = self._resolve_profile(profile_name or pipeline_name, target)

            # Phase 5: Manifest generation
            manifest = self._generate_manifest(pipeline_config, steps, dag, target, resolved_profile)

            # Save manifest to target directory
            self._save_manifest(manifest, pipeline_name)

            return manifest

        except Exception as e:
            raise CompilationError(f"Failed to compile {pipeline_name}: {str(e)}") from e

    def _resolve_composition(self, pipeline_path: Path) -> dict:
        """Phase 1: Resolve pipeline composition.

        Resolves:
        - !include directives
        - base_pipeline inheritance

        Args:
            pipeline_path: Path to pipeline YAML file

        Returns:
            Resolved pipeline dictionary

        Raises:
            CompilationError: If resolution fails
        """
        try:
            # Load with !include support
            yaml_dict = self.composition_resolver.load_with_includes(pipeline_path)

            # Resolve base_pipeline inheritance
            yaml_dict = self.composition_resolver.resolve_base_pipeline(yaml_dict, pipeline_path)

            return yaml_dict
        except Exception as e:
            raise CompilationError(f"Composition resolution failed: {str(e)}") from e

    def _validate_schema(self, yaml_dict: dict) -> PipelineConfig:
        """Phase 2: Validate YAML against Pydantic schema.

        Raises:
            CompilationError: If schema validation fails
        """
        try:
            return PipelineConfig(**yaml_dict)
        except Exception as e:
            raise CompilationError(f"Schema validation failed: {str(e)}") from e

    def _validate_plugins(self, pipeline_config: PipelineConfig):
        """Phase 3: Validate plugin configurations.

        Loads framework plugin and validates framework-specific configuration.

        Raises:
            CompilationError: If plugin validation fails
        """
        # Skip validation for serving pipelines
        config_dict = pipeline_config.model_dump()
        if config_dict.get("serving"):
            return

        # Only validate if we have training configuration
        if not pipeline_config.training:
            return

        # Validate framework plugin
        try:
            from mbt.core.registry import PluginRegistry, MissingAdapterError

            registry = PluginRegistry()

            # Get framework configuration
            framework_name = pipeline_config.training.model_training.framework
            framework_config = pipeline_config.training.model_training.config
            problem_type = pipeline_config.project.problem_type

            # Load framework plugin
            try:
                framework = registry.get("mbt.frameworks", framework_name)
            except MissingAdapterError as e:
                raise CompilationError(
                    f"Framework plugin '{framework_name}' not found.\n"
                    f"Install it with: pip install mbt-{framework_name.replace('_', '-')}"
                ) from e

            # Validate framework configuration
            try:
                framework.validate_config(framework_config, problem_type)
            except ValueError as e:
                raise CompilationError(
                    f"Framework configuration validation failed for '{framework_name}':\n"
                    f"  {str(e)}"
                ) from e

        except CompilationError:
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise CompilationError(f"Plugin validation failed: {str(e)}") from e

    def _build_dag(self, pipeline_config: PipelineConfig) -> tuple:
        """Phase 4: Build DAG from pipeline configuration.

        Returns:
            (steps_dict, dag_definition)
        """
        config_dict = pipeline_config.model_dump()

        # Detect pipeline type
        if pipeline_config.training and not config_dict.get("serving"):
            # Training pipeline
            return self.dag_builder.build_training_dag(config_dict)
        elif config_dict.get("serving"):
            # Serving pipeline
            return self.dag_builder.build_serving_dag(config_dict)
        else:
            raise CompilationError("No training or serving configuration found")

    def _resolve_profile(self, profile_name: str, target: str) -> dict:
        """Resolve profile configuration.

        Args:
            profile_name: Profile name
            target: Target environment

        Returns:
            Resolved profile configuration dict
        """
        try:
            return self.profiles_loader.resolve_target(
                profile_name=profile_name,
                target=target,
                runtime_vars=self.runtime_vars,
            )
        except ProfileNotFoundError:
            # No profile found - return empty dict (use defaults)
            print(f"  ⚠ No profile '{profile_name}' found, using defaults")
            return {}

    def _generate_manifest(
        self,
        pipeline_config: PipelineConfig,
        steps: dict,
        dag,
        target: str,
        profile_config: dict,
    ) -> Manifest:
        """Phase 5: Generate final manifest with all configuration merged."""
        # Detect pipeline type
        config_dict = pipeline_config.model_dump()
        if config_dict.get("serving"):
            pipeline_type = "serving"
        else:
            pipeline_type = "training"

        metadata = ManifestMetadata(
            mbt_version=__version__,
            schema_version=pipeline_config.schema_version,
            generated_at=datetime.utcnow().isoformat() + "Z",
            pipeline_name=pipeline_config.project.name,
            pipeline_type=pipeline_type,
            target=target,
            problem_type=pipeline_config.project.problem_type,
        )

        return Manifest(
            metadata=metadata,
            steps=steps,
            dag=dag,
            profile_config=profile_config,
        )

    def _save_manifest(self, manifest: Manifest, pipeline_name: str):
        """Save manifest to target directory."""
        output_dir = self.target_dir / pipeline_name
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

        print(f"✓ Manifest saved to: {manifest_path}")
