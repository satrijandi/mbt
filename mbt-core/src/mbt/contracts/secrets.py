"""Secrets provider plugin contract.

Secrets providers allow MBT to retrieve sensitive configuration from various backends
(environment variables, AWS Secrets Manager, HashiCorp Vault, etc.).
"""

from abc import ABC, abstractmethod


class SecretsPlugin(ABC):
    """Abstract base class for secrets provider adapters.

    Secrets providers retrieve sensitive configuration values (passwords, API keys,
    tokens) from secure storage backends.

    Example:
        >>> secrets = plugin_registry.get("mbt.secrets", "env")
        >>> db_password = secrets.get_secret("DB_PASSWORD")
    """

    @abstractmethod
    def get_secret(self, key: str) -> str:
        """Retrieve a secret value by key.

        Args:
            key: Secret key/name

        Returns:
            Secret value as string

        Raises:
            KeyError: If secret not found
            RuntimeError: If unable to access secrets backend

        Example:
            >>> password = secrets.get_secret("DB_PASSWORD")
            >>> api_key = secrets.get_secret("OPENAI_API_KEY")
        """
        pass

    def get_secret_with_default(self, key: str, default: str) -> str:
        """Retrieve a secret value with a default fallback.

        Args:
            key: Secret key/name
            default: Default value if secret not found

        Returns:
            Secret value or default

        Note:
            Default implementation calls get_secret() and catches KeyError.
            Override for more efficient implementation.

        Example:
            >>> host = secrets.get_secret_with_default("DB_HOST", "localhost")
        """
        try:
            return self.get_secret(key)
        except KeyError:
            return default

    def validate_access(self) -> bool:
        """Validate that the secrets backend is accessible.

        Returns:
            True if secrets backend is accessible, False otherwise

        Note:
            Default implementation always returns True. Override for backends
            that require connectivity checks.

        Example:
            >>> if not secrets.validate_access():
            ...     print("Warning: Cannot access secrets backend")
        """
        return True

    def list_secret_keys(self) -> list[str]:
        """List all available secret keys.

        Returns:
            List of secret key names

        Note:
            Default implementation raises NotImplementedError. Override if
            backend supports listing keys.

        Raises:
            NotImplementedError: If backend doesn't support listing

        Example:
            >>> keys = secrets.list_secret_keys()
            >>> print(f"Available secrets: {keys}")
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support listing secret keys"
        )
