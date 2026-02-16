"""Configuration loader with Jinja2 templating support.

Supports template functions in profiles.yaml and pipeline YAML:
- {{ env_var('KEY') }} - Read from environment variable
- {{ secret('KEY') }} - Read from secrets provider
- {{ var('KEY') }} - Read from runtime variables (--vars)
"""

import os
from typing import Any
from jinja2 import Template, Environment, StrictUndefined


class ConfigLoader:
    """Loads and renders configuration with Jinja2 templating."""

    def __init__(self, runtime_vars: dict[str, str] | None = None, secrets_provider: Any = None):
        """Initialize config loader.

        Args:
            runtime_vars: Variables passed via CLI (--vars key=value)
            secrets_provider: Secrets provider plugin (optional)
        """
        self.runtime_vars = runtime_vars or {}
        self.secrets_provider = secrets_provider

        # Create Jinja2 environment
        self.jinja_env = Environment(
            undefined=StrictUndefined,  # Error on undefined variables
            autoescape=False,  # Don't escape for YAML
        )

        # Register template functions
        self.jinja_env.globals['env_var'] = self._env_var
        self.jinja_env.globals['secret'] = self._secret
        self.jinja_env.globals['var'] = self._var

    def render_string(self, template_string: str) -> str:
        """Render a template string with Jinja2.

        Args:
            template_string: String containing Jinja2 templates

        Returns:
            Rendered string

        Example:
            >>> loader = ConfigLoader(runtime_vars={"db": "prod_db"})
            >>> loader.render_string("Database: {{ var('db') }}")
            'Database: prod_db'
        """
        template = self.jinja_env.from_string(template_string)
        return template.render()

    def render_dict(self, config_dict: dict) -> dict:
        """Recursively render all string values in a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Dictionary with all template strings rendered

        Example:
            >>> loader = ConfigLoader()
            >>> config = {"uri": "{{ env_var('DB_URI') }}"}
            >>> loader.render_dict(config)
            {"uri": "postgresql://..."}
        """
        result = {}
        for key, value in config_dict.items():
            if isinstance(value, str):
                result[key] = self.render_string(value)
            elif isinstance(value, dict):
                result[key] = self.render_dict(value)
            elif isinstance(value, list):
                result[key] = self._render_list(value)
            else:
                result[key] = value
        return result

    def _render_list(self, config_list: list) -> list:
        """Recursively render all string values in a list."""
        result = []
        for item in config_list:
            if isinstance(item, str):
                result.append(self.render_string(item))
            elif isinstance(item, dict):
                result.append(self.render_dict(item))
            elif isinstance(item, list):
                result.append(self._render_list(item))
            else:
                result.append(item)
        return result

    def _env_var(self, key: str, default: str | None = None) -> str:
        """Template function: Read from environment variable.

        Usage in YAML:
            tracking_uri: "{{ env_var('MLFLOW_TRACKING_URI') }}"
            db_host: "{{ env_var('DB_HOST', 'localhost') }}"

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Environment variable value

        Raises:
            KeyError: If variable not set and no default provided
        """
        value = os.environ.get(key)
        if value is None:
            if default is not None:
                return default
            raise KeyError(f"Environment variable '{key}' not set and no default provided")
        return value

    def _secret(self, key: str) -> str:
        """Template function: Read from secrets provider.

        Usage in YAML:
            password: "{{ secret('DB_PASSWORD') }}"

        Args:
            key: Secret key name

        Returns:
            Secret value

        Raises:
            RuntimeError: If no secrets provider configured
            KeyError: If secret not found
        """
        if self.secrets_provider is None:
            # Fallback to environment variable
            value = os.environ.get(key)
            if value is None:
                raise KeyError(
                    f"Secret '{key}' not found. "
                    "No secrets provider configured, falling back to environment variables."
                )
            return value

        try:
            return self.secrets_provider.get_secret(key)
        except Exception as e:
            raise KeyError(f"Failed to retrieve secret '{key}': {e}")

    def _var(self, key: str, default: str | None = None) -> str:
        """Template function: Read from runtime variables.

        Runtime variables are passed via CLI:
            mbt run --vars execution_date=2026-02-16 run_id=abc123

        Usage in YAML:
            execution_date: "{{ var('execution_date') }}"
            model_run_id: "{{ var('run_id') }}"

        Args:
            key: Variable name
            default: Default value if not set

        Returns:
            Variable value

        Raises:
            KeyError: If variable not set and no default provided
        """
        value = self.runtime_vars.get(key)
        if value is None:
            if default is not None:
                return default
            raise KeyError(
                f"Runtime variable '{key}' not set and no default provided. "
                f"Pass via: --vars {key}=<value>"
            )
        return value
