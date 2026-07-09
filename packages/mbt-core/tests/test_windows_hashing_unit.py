"""Unit tests for window-expression edge cases and environment digests."""

import pytest

from mbt.compile.hashing import env_digest
from mbt.compile.windows import parse_window
from mbt.exceptions import ConfigError


def test_bare_iso_timestamp_is_not_a_window() -> None:
    """A lone timestamp contains ':' but is no '<start>:<end>' expression."""
    with pytest.raises(ConfigError, match="invalid window expression") as excinfo:
        parse_window("2026-01-01T00:00:00")
    assert "bare duration" in (excinfo.value.hint or "")


def test_env_digest_tolerates_uninstalled_fingerprint_packages() -> None:
    digest = env_digest(["mbt-definitely-not-installed-zzz"])
    assert digest.startswith("sha256:")
    assert digest == env_digest(["mbt-definitely-not-installed-zzz"])  # deterministic
    assert digest != env_digest([])  # the '(not installed)' line still counts
