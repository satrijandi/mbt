"""Environment variable secrets provider.

Default secrets provider that reads secrets from environment variables.
"""

import os
from mbt.contracts.secrets import SecretsPlugin


class EnvSecretsProvider(SecretsPlugin):
    """Secrets provider that reads from environment variables.

    This is the default secrets provider. It's simple and works everywhere,
    but is less secure than dedicated secrets management systems.

    For production, consider using:
    - AWS Secrets Manager (mbt-aws-secrets)
    - HashiCorp Vault (mbt-vault-secrets)
    - Google Secret Manager (mbt-gcp-secrets)

    Example:
        >>> secrets = EnvSecretsProvider()
        >>> db_password = secrets.get_secret("DB_PASSWORD")
    """

    def get_secret(self, key: str) -> str:
        """Read secret from environment variable.

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            KeyError: If environment variable not set

        Example:
            >>> os.environ["API_KEY"] = "secret123"
            >>> secrets = EnvSecretsProvider()
            >>> secrets.get_secret("API_KEY")
            'secret123'
        """
        value = os.environ.get(key)
        if value is None:
            raise KeyError(
                f"Secret '{key}' not found in environment variables. "
                f"Set it with: export {key}=<value>"
            )
        return value

    def validate_access(self) -> bool:
        """Environment variables are always accessible.

        Returns:
            Always True
        """
        return True

    def list_secret_keys(self) -> list[str]:
        """List all environment variables.

        Returns:
            List of all environment variable names

        Example:
            >>> secrets = EnvSecretsProvider()
            >>> keys = secrets.list_secret_keys()
            >>> "PATH" in keys
            True
        """
        return list(os.environ.keys())
