"""FEEDBACK v7 B-1: every Jinja environment mbt builds is sandboxed.

``SECURITY.md`` says YAML and Jinja are not code. ``profiles.yml`` (committed,
rendered on every pull request) and the job's adapter-config re-render built a
plain ``jinja2.Environment`` and so were: ``{{ lipsum.__globals__['os'] }}``
reached the ``os`` module. The first test pins the behaviour at each site; the
second keeps a fourth site from appearing.
"""

import re
from pathlib import Path

import pytest
from jinja2.sandbox import SecurityError

from mbt.config.profiles import load_profiles
from mbt.exceptions import ConfigError

PACKAGES = Path(__file__).resolve().parents[2]

#: The hostile expression: through ``lipsum`` (a function every environment
#: exposes) to its module globals, to ``os``. Harmless to run, which is the point.
ESCAPE = "{{ lipsum.__globals__['os'].getpid() }}"


def test_profiles_yml_cannot_reach_the_os_module(tmp_path: Path) -> None:
    (tmp_path / "profiles.yml").write_text(
        "p:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      data: {adapter: local, config: {root: .}}\n"
        "      compute: {adapter: local}\n"
        f'      vars: {{probe: "{ESCAPE}"}}\n'
    )
    with pytest.raises(ConfigError, match=r"invalid Jinja in profiles\.yml.*unsafe"):
        load_profiles("p", tmp_path, profiles_dir=tmp_path)


def test_the_jobs_adapter_config_render_cannot_reach_the_os_module() -> None:
    from mbt.execute.job_setup import _render_adapter_ref
    from mbt_adapter_base import AdapterRef

    ref = AdapterRef(adapter="local", config={"root": ESCAPE})
    with pytest.raises(SecurityError, match="unsafe"):
        _render_adapter_ref(ref, {})


def test_no_source_file_builds_an_unsandboxed_environment() -> None:
    pattern = re.compile(
        r"\bjinja2\.Environment\(|\bjinja2\.Template\("
        r"|from jinja2 import[^\n]*\b(Environment|Template)\b"
        r"|\bNativeEnvironment\("
    )
    offenders = [
        f"{path.relative_to(PACKAGES)}:{number}"
        for path in sorted(PACKAGES.glob("*/src/**/*.py"))
        if "_scaffold" not in path.parts
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "build Jinja environments through mbt.jinja.sandbox, never directly "
        f"(FEEDBACK v7 B-1): {offenders}"
    )
