"""The showcase tier's gates hold, checked where they matter (SHOW-15).

Every `test_showcase_*` module that needs the docker stack is opt-in: it skips
unless MBT_LIVE_SHOWCASE=1, and once opted in a missing docker FAILS instead of
skipping. That contract is only worth anything in the suites that never opt in,
so it is checked here, hermetically, in the ordinary fast suite - where a module
that lost its gate would otherwise try to boot a whole platform stack.

The gated modules are discovered by filename rather than listed. A hand-kept
list is what this check used to be, and it went stale twice: the object-store
and warehouse modules were added after it and were never checked.
"""

import importlib
import os
from pathlib import Path

import pytest
import showcase_utils
from showcase_utils import SKIP_REASON, SNOWFLAKE_SKIP_REASON

TESTS_DIR = Path(__file__).resolve().parent

#: Showcase modules that need no stack and run in the fast suite. Everything
#: else matching test_showcase_*.py is gated.
HERMETIC = frozenset(
    {
        "test_showcase_gates",
        "test_showcase_image_pins",
        "test_showcase_seaweedfs_plane",
        "test_showcase_wide_scripts",
    }
)

#: Gated modules that carry a second opt-in on top of MBT_LIVE_SHOWCASE=1,
#: and the variable their skip reason must name. Dropping one of these would
#: put the k3d tier on the nightly path (no k3d there), boot a second full
#: stack beside the session one, or send warehouse traffic from docker alone.
EXTRA_GATES = {
    "test_showcase_k3d": "MBT_LIVE_SHOWCASE_K3D=1",
    "test_showcase_make": "MBT_LIVE_SHOWCASE_MAKE=1",
    "test_showcase_snowflake": "MBT_LIVE_SNOWFLAKE=1",
}


def _module_names() -> set[str]:
    return {path.stem for path in TESTS_DIR.glob("test_showcase_*.py")}


def _skipifs(module_name: str) -> list[pytest.MarkDecorator]:
    marks = importlib.import_module(module_name).pytestmark
    return [mark for mark in marks if mark.name == "skipif"]


def test_the_module_lists_name_modules_that_exist() -> None:
    names = _module_names()
    assert names >= HERMETIC, f"HERMETIC names missing modules: {sorted(HERMETIC - names)}"
    assert names >= set(EXTRA_GATES), f"EXTRA_GATES names missing modules: {EXTRA_GATES}"


def test_every_gated_showcase_module_keeps_the_opt_in_gate() -> None:
    gated = sorted(_module_names() - HERMETIC)
    assert gated, "no gated showcase modules found"
    for name in gated:
        module = importlib.import_module(name)
        marks = getattr(module, "pytestmark", None)
        assert isinstance(marks, list), f"{name} has no pytestmark list"
        mark_names = {mark.name for mark in marks}
        assert {"live", "live_showcase"} <= mark_names, (name, sorted(mark_names))

        gate = [mark for mark in _skipifs(name) if mark.kwargs.get("reason") == SKIP_REASON]
        assert len(gate) == 1, f"{name} lost the MBT_LIVE_SHOWCASE opt-in skipif"
        # The condition is evaluated at import, so compare it with what the
        # environment says right now: a constant (True/False) would pass the
        # reason check and still never, or always, skip.
        assert gate[0].args[0] is (os.environ.get("MBT_LIVE_SHOWCASE") != "1"), (
            f"{name}'s skipif is not keyed on MBT_LIVE_SHOWCASE"
        )


def test_the_extra_gated_modules_keep_their_second_gate() -> None:
    for name, variable in EXTRA_GATES.items():
        reasons = [mark.kwargs.get("reason") or "" for mark in _skipifs(name)]
        assert any(variable in reason for reason in reasons), (
            f"{name} lost the skipif naming {variable}: {reasons}"
        )


def test_hermetic_showcase_modules_are_not_gated() -> None:
    """The other direction: marking one of these live would silently drop it
    out of the fast suite, which is the only place it runs."""
    for name in sorted(HERMETIC - {"test_showcase_gates"}):
        module = importlib.import_module(name)
        assert not getattr(module, "pytestmark", None), f"{name} gained a module-level mark"


def test_opting_in_without_docker_fails_instead_of_skipping(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(pytest.fail.Exception, match="docker is not on PATH"):
        showcase_utils.require_docker()


def test_opting_in_with_an_unreachable_daemon_fails(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "Cannot connect"),
    )
    with pytest.raises(pytest.fail.Exception, match="daemon is not reachable"):
        showcase_utils.require_docker()


def test_the_snowflake_gate_fails_on_incomplete_credentials(monkeypatch) -> None:
    """Gate 3 of the warehouse plane: opted in but misconfigured must FAIL."""
    assert "MBT_LIVE_SNOWFLAKE=1" in SNOWFLAKE_SKIP_REASON
    for name in (*showcase_utils.SNOWFLAKE_REQUIRED_ENV, *showcase_utils.SNOWFLAKE_AUTH_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(pytest.fail.Exception, match="SNOWFLAKE_ACCOUNT"):
        showcase_utils.require_snowflake()

    for name in showcase_utils.SNOWFLAKE_REQUIRED_ENV:
        monkeypatch.setenv(name, "set")
    with pytest.raises(pytest.fail.Exception, match="no auth is configured"):
        showcase_utils.require_snowflake()
