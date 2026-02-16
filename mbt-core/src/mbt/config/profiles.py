"""Profiles configuration loader.

Profiles define environment-specific configuration (dev, staging, prod) including:
- Data connectors
- Storage backends
- Executor configuration
- MLflow tracking URIs
- Secrets providers
"""

import yaml
from pathlib import Path
from typing import Any
import os

from mbt.config.loader import ConfigLoader


class ProfileNotFoundError(Exception):
    """Raised when a profile is not found."""
    pass


class ProfilesLoader:
    """Loads and resolves profiles.yaml configuration.

    Profiles can be located in:
    1. Project root: ./profiles.yaml
    2. User home: ~/.mbt/profiles.yaml
    3. Explicit path via MBT_PROFILES_PATH env var

    Example profiles.yaml:
        my-project:
          target: dev

          outputs:
            dev:
              executor:
                type: local
              storage:
                type: local
                config:
                  base_path: ./local_artifacts

            prod:
              executor:
                type: kubernetes
              storage:
                type: s3
                config:
                  bucket: my-bucket
    """

    def __init__(self, project_root: Path):
        """Initialize profiles loader.

        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root)
        self._profiles_cache = None

    def load_profiles(self) -> dict[str, Any]:
        """Load profiles from profiles.yaml.

        Search order:
        1. MBT_PROFILES_PATH environment variable
        2. ./profiles.yaml (project root)
        3. ~/.mbt/profiles.yaml (user home)

        Returns:
            Dictionary of profiles

        Raises:
            FileNotFoundError: If profiles.yaml not found
        """
        if self._profiles_cache is not None:
            return self._profiles_cache

        # Search for profiles.yaml
        profiles_path = self._find_profiles_file()

        if profiles_path is None:
            # No profiles.yaml found - return empty dict (use defaults)
            return {}

        # Load and parse YAML
        with open(profiles_path) as f:
            profiles = yaml.safe_load(f)

        self._profiles_cache = profiles or {}
        return self._profiles_cache

    def _find_profiles_file(self) -> Path | None:
        """Find profiles.yaml file.

        Returns:
            Path to profiles.yaml or None if not found
        """
        # 1. Check MBT_PROFILES_PATH env var
        env_path = os.environ.get("MBT_PROFILES_PATH")
        if env_path:
            env_path = Path(env_path)
            if env_path.exists():
                return env_path

        # 2. Check project root
        project_profiles = self.project_root / "profiles.yaml"
        if project_profiles.exists():
            return project_profiles

        # 3. Check user home
        home_profiles = Path.home() / ".mbt" / "profiles.yaml"
        if home_profiles.exists():
            return home_profiles

        return None

    def get_profile(self, profile_name: str) -> dict[str, Any]:
        """Get a specific profile by name.

        Args:
            profile_name: Name of the profile

        Returns:
            Profile configuration dictionary

        Raises:
            ProfileNotFoundError: If profile not found
        """
        profiles = self.load_profiles()

        if profile_name not in profiles:
            raise ProfileNotFoundError(
                f"Profile '{profile_name}' not found in profiles.yaml. "
                f"Available profiles: {list(profiles.keys())}"
            )

        return profiles[profile_name]

    def resolve_target(
        self,
        profile_name: str,
        target: str | None = None,
        runtime_vars: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Resolve target configuration from profile.

        Args:
            profile_name: Name of the profile
            target: Target environment (dev, staging, prod)
                   If None, uses default from profile
            runtime_vars: Runtime variables for template rendering

        Returns:
            Merged configuration for the target

        Raises:
            ProfileNotFoundError: If profile or target not found

        Example:
            >>> loader = ProfilesLoader(Path("/project"))
            >>> config = loader.resolve_target("my-project", "prod")
            >>> config["storage"]["type"]
            's3'
        """
        profile = self.get_profile(profile_name)

        # Determine target
        if target is None:
            target = profile.get("target", "dev")

        # Get target-specific configuration
        if "outputs" not in profile:
            raise ProfileNotFoundError(
                f"Profile '{profile_name}' has no 'outputs' section"
            )

        if target not in profile["outputs"]:
            raise ProfileNotFoundError(
                f"Target '{target}' not found in profile '{profile_name}'. "
                f"Available targets: {list(profile['outputs'].keys())}"
            )

        target_config = profile["outputs"][target]

        # Merge profile-level config with target-specific config
        merged_config = self._merge_config(profile, target_config)

        # Render templates
        config_loader = ConfigLoader(runtime_vars=runtime_vars)
        rendered_config = config_loader.render_dict(merged_config)

        return rendered_config

    def _merge_config(self, profile: dict, target_config: dict) -> dict:
        """Merge profile-level config with target-specific config.

        Target-specific config overrides profile-level config.

        Args:
            profile: Profile-level configuration
            target_config: Target-specific configuration

        Returns:
            Merged configuration
        """
        merged = {}

        # Copy profile-level config (excluding 'outputs' and 'target')
        for key, value in profile.items():
            if key not in ["outputs", "target"]:
                merged[key] = value

        # Merge target-specific config (overrides profile-level)
        for key, value in target_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Deep merge dictionaries
                merged[key] = {**merged[key], **value}
            else:
                # Override
                merged[key] = value

        return merged
