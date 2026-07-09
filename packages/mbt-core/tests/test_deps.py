"""`mbt deps` install semantics (FR-PROJ-04; FEEDBACK 2.2: loose installs
had no lock path and no verification)."""

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from mbt.deps import PackagePin, install_packages, verify_pins
from mbt.exceptions import ConfigError

PINS = [PackagePin(package="mbt-xgboost", version="~=0.1")]


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def emit(self, event: object) -> None:
        self.messages.append(event)


@pytest.fixture()
def recorded_bus(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr("mbt.deps.get_bus", lambda: recorder)
    return recorder


@pytest.fixture()
def pip_calls(monkeypatch) -> dict:
    calls: dict = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mbt.deps.subprocess.run", fake_run)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.1.4")
    return calls


def test_pinned_requirements_file_is_preferred(
    tmp_path: Path, pip_calls: dict, recorded_bus: _Recorder
) -> None:
    pinned = tmp_path / "requirements.txt"
    pinned.write_text("mbt-xgboost==0.1.4\n")
    install_packages(PINS, requirements_file=pinned)
    assert pip_calls["command"][-2:] == ["-r", str(pinned)]
    assert any(
        "pinned requirements" in str(getattr(m, "message", "")) for m in recorded_bus.messages
    )


def test_loose_fallback_installs_specifiers_and_warns(
    pip_calls: dict, recorded_bus: _Recorder
) -> None:
    install_packages(PINS)
    assert pip_calls["command"][-1] == "mbt-xgboost~=0.1"
    warns = [m for m in recorded_bus.messages if getattr(m, "level", "") == "warn"]
    assert warns and "unpinned" in str(warns[0].message)


def test_post_install_verification_catches_drift(
    tmp_path: Path, monkeypatch, recorded_bus: _Recorder
) -> None:
    """A requirements.txt that drifted from packages.yml fails loudly at
    install time, not at the first import."""
    monkeypatch.setattr(
        "mbt.deps.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.9.0")  # outside ~=0.1
    pinned = tmp_path / "requirements.txt"
    pinned.write_text("mbt-xgboost==9.9.0\n")
    with pytest.raises(ConfigError, match="does not satisfy"):
        install_packages(PINS, requirements_file=pinned)


def test_verify_pins_reports_missing_packages(monkeypatch) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    with pytest.raises(ConfigError, match="is not installed"):
        verify_pins(PINS)


def test_dry_run_never_calls_pip(monkeypatch) -> None:
    def exploding_run(command, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("pip must not be invoked on --dry-run")

    monkeypatch.setattr("mbt.deps.subprocess.run", exploding_run)
    assert install_packages(PINS, dry_run=True) == ["mbt-xgboost~=0.1"]
