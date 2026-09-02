"""Unit tests for mbt.jinja.environment: macros, var scopes, error translation."""

from pathlib import Path

import pytest
from core_helpers import write

from mbt.exceptions import CompilationError, ConfigError
from mbt.jinja.environment import ResolveContext, SpecRenderer, TargetContext
from mbt.secrets import clear_taints, redact


def make_ctx(**kwargs) -> ResolveContext:
    defaults = {
        "target": TargetContext(name="dev", threads=2),
        "cli_vars": {},
        "target_vars": {},
        "project_vars": {},
        "ref_resolver": lambda name: f"dataset.p.{name}",
        "source_resolver": lambda group, table: f"source.p.{group}.{table}",
    }
    defaults.update(kwargs)
    return ResolveContext(**defaults)


def test_target_context_renders_as_its_name() -> None:
    assert str(TargetContext(name="prod")) == "prod"


def test_lookup_var_precedence_and_error() -> None:
    ctx = make_ctx(
        cli_vars={"a": 1},
        target_vars={"a": 2, "b": 2},
        project_vars={"b": 3, "c": 3},
    )
    assert ctx.lookup_var("a") == 1  # CLI wins
    assert ctx.lookup_var("b") == 2  # target beats project
    assert ctx.lookup_var("c") == 3
    assert ctx.lookup_var("d", 4) == 4  # explicit default
    with pytest.raises(CompilationError, match="var 'd' has no value and no default"):
        ctx.lookup_var("d")


# -- macros -----------------------------------------------------------------------


def test_macros_load_and_render(tmp_path: Path) -> None:
    write(
        tmp_path / "macros/helpers.jinja",
        """
        {% macro greet(name) %}hello {{ name }}{% endmacro %}
        {% macro _hidden() %}nope{% endmacro %}
        {% set answer = 42 %}
        """,
    )
    renderer = SpecRenderer(macro_paths=[tmp_path / "macros", tmp_path / "missing_dir"])
    assert renderer.macro_names == ["greet"]  # _hidden and non-callables skipped
    captured = renderer.capture(
        {"msg": "{{ greet('world') }}"}, resource="r", path=tmp_path / "spec.yml"
    )
    assert captured.rendered["msg"] == "hello world"


def test_invalid_macro_file_raises_config_error(tmp_path: Path) -> None:
    write(tmp_path / "macros/broken.jinja", "{% macro oops(\n")
    with pytest.raises(ConfigError, match="invalid macro file"):
        SpecRenderer(macro_paths=[tmp_path / "macros"])


# -- capture phase ------------------------------------------------------------------


def test_capture_var_scopes_and_default(tmp_path: Path) -> None:
    renderer = SpecRenderer()
    captured = renderer.capture(
        {
            "from_cli": "{{ var('x') }}",
            "from_project": "{{ var('y') }}",
            "from_default": "{{ var('z', 9) }}",
        },
        resource="r",
        path=tmp_path / "spec.yml",
        cli_vars={"x": 1, "y": 0},
        project_vars={"y": 2},
    )
    assert captured.rendered == {"from_cli": 1, "from_project": 0, "from_default": 9}


def test_capture_var_without_value_raises(tmp_path: Path) -> None:
    renderer = SpecRenderer()
    with pytest.raises(ConfigError, match="var 'nope' has no value at parse time"):
        renderer.capture({"v": "{{ var('nope') }}"}, resource="r", path=tmp_path / "s.yml")


def test_capture_env_var_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MBT_UNSET_ZZZ", raising=False)
    monkeypatch.setenv("MBT_SET_ZZZ", "present")
    renderer = SpecRenderer()
    captured = renderer.capture(
        {
            "set": "{{ env_var('MBT_SET_ZZZ') }}",
            "defaulted": "{{ env_var('MBT_UNSET_ZZZ', 'fallback') }}",
            "empty": "{{ env_var('MBT_UNSET_ZZZ') }}",
        },
        resource="r",
        path=tmp_path / "s.yml",
    )
    assert captured.rendered == {"set": "present", "defaulted": "fallback", "empty": ""}


def test_capture_env_defaults(tmp_path: Path, monkeypatch) -> None:
    """`env()` is `env_var()` minus the taint, so its branches must match."""
    monkeypatch.delenv("MBT_UNSET_ZZZ", raising=False)
    monkeypatch.setenv("MBT_SET_ZZZ", "present")
    renderer = SpecRenderer()
    captured = renderer.capture(
        {
            "set": "{{ env('MBT_SET_ZZZ') }}",
            "defaulted": "{{ env('MBT_UNSET_ZZZ', 'fallback') }}",
            "empty": "{{ env('MBT_UNSET_ZZZ') }}",
        },
        resource="r",
        path=tmp_path / "s.yml",
    )
    assert captured.rendered == {"set": "present", "defaulted": "fallback", "empty": ""}


# -- resolve phase --------------------------------------------------------------------


def test_resolve_env_var_branches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MBT_UNSET_ZZZ", raising=False)
    monkeypatch.setenv("MBT_SET_ZZZ", "present")
    renderer = SpecRenderer()
    ctx = make_ctx()
    rendered = renderer.resolve(
        {
            "set": "{{ env_var('MBT_SET_ZZZ') }}",
            "defaulted": "{{ env_var('MBT_UNSET_ZZZ', 'fallback') }}",
        },
        ctx,
        resource="r",
        path=tmp_path / "s.yml",
    )
    assert rendered == {"set": "present", "defaulted": "fallback"}
    with pytest.raises(CompilationError, match="environment variable 'MBT_UNSET_ZZZ' is not set"):
        renderer.resolve(
            {"v": "{{ env_var('MBT_UNSET_ZZZ') }}"}, ctx, resource="r", path=tmp_path / "s.yml"
        )


def test_resolve_env_branches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MBT_UNSET_ZZZ", raising=False)
    monkeypatch.setenv("MBT_SET_ZZZ", "present")
    renderer = SpecRenderer()
    ctx = make_ctx()
    rendered = renderer.resolve(
        {
            "set": "{{ env('MBT_SET_ZZZ') }}",
            "defaulted": "{{ env('MBT_UNSET_ZZZ', 'fallback') }}",
        },
        ctx,
        resource="r",
        path=tmp_path / "s.yml",
    )
    assert rendered == {"set": "present", "defaulted": "fallback"}
    with pytest.raises(CompilationError, match="environment variable 'MBT_UNSET_ZZZ' is not set"):
        renderer.resolve(
            {"v": "{{ env('MBT_UNSET_ZZZ') }}"}, ctx, resource="r", path=tmp_path / "s.yml"
        )


def test_env_does_not_taint_but_env_var_does(tmp_path: Path, monkeypatch) -> None:
    """The whole point of the split (FEEDBACK v3 A-1).

    Redaction is exact-substring, so tainting a short non-secret rewrites
    unrelated text - including the floats in run_results.json. `env_var()`
    must still taint (it is the credential path); `env()` must not.
    """
    monkeypatch.setenv("MBT_PORT_ZZZ", "1")
    renderer = SpecRenderer()
    ctx = make_ctx()
    payload = '{"pr_auc":0.1234}'

    clear_taints()
    renderer.resolve(
        {"v": "{{ env_var('MBT_PORT_ZZZ') }}"}, ctx, resource="r", path=tmp_path / "s.yml"
    )
    assert redact(payload) == '{"pr_auc":0.***234}'

    clear_taints()
    renderer.resolve({"v": "{{ env('MBT_PORT_ZZZ') }}"}, ctx, resource="r", path=tmp_path / "s.yml")
    assert redact(payload) == payload


def test_resolve_renders_refs_sources_and_target(tmp_path: Path) -> None:
    renderer = SpecRenderer()
    rendered = renderer.resolve(
        {
            "dataset": "ref('churn')",
            "table": "source('lake', 'subs')",
            "env": "{{ target.name }}",
            "nested": {"list": ["{{ var('n', 400) }}"]},
        },
        make_ctx(),
        resource="r",
        path=tmp_path / "s.yml",
    )
    assert rendered["dataset"] == "dataset.p.churn"
    assert rendered["table"] == "source.p.lake.subs"
    assert rendered["env"] == "dev"
    assert rendered["nested"]["list"] == [400]  # native types survive


# -- error translation ----------------------------------------------------------------


def test_undefined_name_is_a_config_error_at_parse(tmp_path: Path) -> None:
    # NOTE: the template must force string concatenation; a lone "{{ typo }}"
    # renders to the StrictUndefined object itself in the native environment.
    renderer = SpecRenderer()
    with pytest.raises(ConfigError, match="undefined Jinja name") as excinfo:
        renderer.capture({"v": "{{ nonexistent_thing }}!"}, resource="r", path=tmp_path / "s.yml")
    assert "available: ref, source, var" in (excinfo.value.hint or "")


def test_undefined_name_is_a_compilation_error_at_compile(tmp_path: Path) -> None:
    renderer = SpecRenderer()
    with pytest.raises(CompilationError, match="undefined Jinja name"):
        renderer.resolve(
            {"v": "{{ nonexistent_thing }}!"}, make_ctx(), resource="r", path=tmp_path / "s.yml"
        )


def test_jinja_syntax_error_is_translated(tmp_path: Path) -> None:
    renderer = SpecRenderer()
    with pytest.raises(ConfigError, match="invalid Jinja"):
        renderer.capture({"v": "{% if %}"}, resource="r", path=tmp_path / "s.yml")
    with pytest.raises(CompilationError, match="invalid Jinja"):
        renderer.resolve({"v": "{% if %}"}, make_ctx(), resource="r", path=tmp_path / "s.yml")
