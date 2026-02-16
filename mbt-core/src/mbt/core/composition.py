"""Pipeline composition resolver.

Handles:
- base_pipeline: Single-level inheritance
- !include: YAML fragment inclusion
"""

import yaml
from pathlib import Path
from typing import Any


class IncludeConstructor:
    """YAML constructor for !include directive."""

    def __init__(self, base_path: Path):
        """Initialize include constructor.

        Args:
            base_path: Base path for resolving relative includes
        """
        self.base_path = base_path

    def __call__(self, loader: yaml.Loader, node: yaml.Node) -> Any:
        """Construct value for !include tag.

        Args:
            loader: YAML loader
            node: YAML node

        Returns:
            Loaded content from included file
        """
        # Get the file path from the node
        include_path = loader.construct_scalar(node)

        # Resolve relative to base path
        full_path = self.base_path / include_path

        if not full_path.exists():
            raise FileNotFoundError(f"Include file not found: {full_path}")

        # Load the included file
        with open(full_path) as f:
            return yaml.safe_load(f)


class CompositionResolver:
    """Resolves pipeline composition (base_pipeline and !include)."""

    def __init__(self, pipelines_dir: Path, runtime_vars: dict[str, str] | None = None):
        """Initialize composition resolver.

        Args:
            pipelines_dir: Directory containing pipeline YAML files
            runtime_vars: Runtime variables for template rendering
        """
        self.pipelines_dir = Path(pipelines_dir)
        self.runtime_vars = runtime_vars or {}

        # Create ConfigLoader for template rendering
        from mbt.config.loader import ConfigLoader
        self.config_loader = ConfigLoader(runtime_vars=self.runtime_vars)

    def load_with_includes(self, yaml_path: Path) -> dict:
        """Load YAML file with !include support.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Loaded YAML with includes resolved

        Example:
            YAML file:
                schema: !include ../includes/schema.yaml

            Result:
                schema:
                  target:
                    label_column: churned
                  ...
        """
        # Register !include constructor
        include_constructor = IncludeConstructor(yaml_path.parent)
        yaml.add_constructor('!include', include_constructor, Loader=yaml.SafeLoader)

        # Load YAML
        with open(yaml_path) as f:
            yaml_dict = yaml.safe_load(f)

        # Render templates
        yaml_dict = self.config_loader.render_dict(yaml_dict)

        return yaml_dict

    def resolve_base_pipeline(self, pipeline_dict: dict, pipeline_path: Path) -> dict:
        """Resolve base_pipeline inheritance.

        Single-level inheritance with deep merge semantics:
        - Child overrides parent for scalar values
        - Child extends parent for lists
        - Child deep-merges parent for dicts

        Args:
            pipeline_dict: Pipeline configuration dictionary
            pipeline_path: Path to current pipeline file (for resolving relative base paths)

        Returns:
            Merged pipeline configuration

        Example:
            Base pipeline (_base_churn.yaml):
                project:
                  problem_type: binary_classification
                training:
                  evaluation:
                    primary_metric: roc_auc

            Child pipeline (churn_h2o_v1.yaml):
                base_pipeline: _base_churn
                project:
                  name: churn_h2o_v1
                training:
                  model_training:
                    framework: h2o_automl

            Result:
                project:
                  name: churn_h2o_v1  # Child override
                  problem_type: binary_classification  # From base
                training:
                  evaluation:
                    primary_metric: roc_auc  # From base
                  model_training:
                    framework: h2o_automl  # From child
        """
        if "base_pipeline" not in pipeline_dict:
            return pipeline_dict

        base_pipeline_name = pipeline_dict["base_pipeline"]

        # Load base pipeline
        base_path = self.pipelines_dir / f"{base_pipeline_name}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base pipeline not found: {base_path}\n"
                f"Referenced in: {pipeline_path}"
            )

        # Load base pipeline with includes
        base_dict = self.load_with_includes(base_path)

        # Remove base_pipeline key from child (not part of schema)
        child_dict = {k: v for k, v in pipeline_dict.items() if k != "base_pipeline"}

        # Deep merge: base + child
        merged = self._deep_merge(base_dict, child_dict)

        return merged

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary

        Merge rules:
        - Scalar: override wins
        - List: override wins (no append/extend)
        - Dict: recursive merge
        """
        result = base.copy()

        for key, value in override.items():
            if key in result:
                # Key exists in both
                base_value = result[key]

                if isinstance(base_value, dict) and isinstance(value, dict):
                    # Both are dicts: recursive merge
                    result[key] = self._deep_merge(base_value, value)
                else:
                    # Override wins (scalar or list)
                    result[key] = value
            else:
                # Key only in override
                result[key] = value

        return result
