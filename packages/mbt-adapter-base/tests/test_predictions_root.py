"""Unit tests for resolve_predictions_root - the staging-root default (F20)."""

import tempfile
from pathlib import Path

from mbt_adapter_base.predictions import resolve_predictions_root


def test_resolve_predictions_root_uses_the_configured_path() -> None:
    assert resolve_predictions_root("/data/preds") == Path("/data/preds")


def test_resolve_predictions_root_defaults_off_project_to_the_tempdir() -> None:
    # Unset -> an absolute, non-project location under the OS temp dir; a
    # scheduled scoring run must never write prediction runs into its checkout.
    root = resolve_predictions_root(None)
    assert root == Path(tempfile.gettempdir()) / "mbt-predictions"
    assert root.is_absolute()
    assert root != Path(".")


def test_resolve_predictions_root_treats_empty_string_as_unset() -> None:
    # "" is a degenerate config (it would resolve to cwd); fall back to the
    # safe default rather than stage into the project tree.
    assert resolve_predictions_root("") == Path(tempfile.gettempdir()) / "mbt-predictions"
