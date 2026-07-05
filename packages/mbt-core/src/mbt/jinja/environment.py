"""The shared Jinja environment and its two rendering phases (TSD §6).

1. **Capture phase (parse):** ``ref``/``source`` record DAG edges and render
   to themselves; ``var``/``env_var`` return inert placeholders, so parsing
   needs neither profiles nor environment.
2. **Resolve phase (compile):** full rendering against the selected target;
   ``ref``/``source`` render to unique_ids, ``var``/``env_var`` to values.

Rendering uses a sandboxed *native* environment so ``{{ var('n', 400) }}``
stays an int instead of becoming the string ``"400"``.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
from jinja2.nativetypes import NativeCodeGenerator, NativeTemplate, native_concat
from jinja2.sandbox import SandboxedEnvironment

from mbt.contracts import AUTO
from mbt.exceptions import CompilationError, ConfigError
from mbt.secrets import taint

#: A field value that is a bare ref()/source() call is sugar for {{ ... }}.
_BARE_CALL_RE = re.compile(r"^\s*(ref|source)\((?P<args>[^)]*)\)\s*$")
_JINJA_MARKERS = ("{{", "{%")

_MISSING = object()


class _NativeSandbox(SandboxedEnvironment):
    """Sandboxed environment that preserves native Python types."""

    code_generator_class = NativeCodeGenerator
    template_class = NativeTemplate
    concat = staticmethod(native_concat)  # type: ignore[assignment]


@dataclass
class CaptureResult:
    """Capture-phase output: rendered mapping plus recorded edges.

    The rendered mapping resolves ``var()`` against CLI + project vars (the
    scopes available without profiles) so schema validation can run at parse
    time; target-scoped vars need a project-level default to parse.
    """

    rendered: dict[str, Any] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class TargetContext:
    """Non-secret target facts exposed as ``{{ target }}`` (TSD §6)."""

    name: str
    threads: int = 1

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ResolveContext:
    """Everything the resolve phase needs to substitute real values."""

    target: TargetContext
    cli_vars: dict[str, Any]
    target_vars: dict[str, Any]
    project_vars: dict[str, Any]
    ref_resolver: Callable[[str], str]  # name -> unique_id (raises on unknown)
    source_resolver: Callable[[str, str], str]  # (group, table) -> unique_id

    def lookup_var(self, name: str, default: Any = _MISSING) -> Any:
        """--vars CLI > target vars > project vars > default (TSD §6)."""
        for scope in (self.cli_vars, self.target_vars, self.project_vars):
            if name in scope:
                return scope[name]
        if default is _MISSING:
            raise CompilationError(
                f"var {name!r} has no value and no default",
                hint=(
                    f"pass --vars '{name}: <value>', set it in the target's vars, "
                    "or in mbt_project.yml vars"
                ),
            )
        return default


def _normalize_template(value: str) -> str | None:
    """Return the template text to render, or None when no Jinja is present."""
    if _BARE_CALL_RE.match(value):
        return "{{ " + value.strip() + " }}"
    if any(marker in value for marker in _JINJA_MARKERS):
        return value
    return None


class SpecRenderer:
    """Renders spec mappings in capture or resolve phase, plus macros."""

    def __init__(self, macro_paths: list[Path] | None = None) -> None:
        self._env = _NativeSandbox(undefined=jinja2.StrictUndefined, autoescape=False)  # noqa: S701
        self._macro_names: list[str] = []
        for macro_dir in macro_paths or []:
            if macro_dir.is_dir():
                for macro_file in sorted(macro_dir.glob("*.jinja")):
                    self._load_macro_file(macro_file)

    @property
    def macro_names(self) -> list[str]:
        return list(self._macro_names)

    def _load_macro_file(self, macro_file: Path) -> None:
        try:
            module = self._env.from_string(macro_file.read_text()).module
        except jinja2.TemplateError as exc:
            raise ConfigError(
                f"invalid macro file: {exc}", path=macro_file, hint="fix the Jinja syntax"
            ) from exc
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj):
                self._env.globals[name] = obj
                self._macro_names.append(name)

    # -- capture phase -----------------------------------------------------

    def capture(
        self,
        mapping: dict[str, Any],
        *,
        resource: str,
        path: Path,
        cli_vars: dict[str, Any] | None = None,
        project_vars: dict[str, Any] | None = None,
    ) -> CaptureResult:
        """Render every string value with edge-capturing context functions."""
        result = CaptureResult()
        cli_scope = cli_vars or {}
        project_scope = project_vars or {}

        def ref(name: str) -> str:
            result.refs.append(str(name))
            return f"ref('{name}')"

        def source(group: str, table: str) -> str:
            result.sources.append((str(group), str(table)))
            return f"source('{group}', '{table}')"

        def var(name: str, default: Any = _MISSING) -> Any:
            for scope in (cli_scope, project_scope):
                if name in scope:
                    return scope[name]
            if default is _MISSING:
                raise ConfigError(
                    f"var {name!r} has no value at parse time",
                    resource=resource,
                    path=path,
                    hint=(
                        "give it a project-level default in mbt_project.yml vars "
                        "(target vars can still override it at compile time) or pass --vars"
                    ),
                )
            return default

        def env_var(name: str, default: str | None = None) -> str:
            value = os.environ.get(name)
            if value is not None:
                return taint(value)
            return default if default is not None else ""

        context = {
            "ref": ref,
            "source": source,
            "var": var,
            "env_var": env_var,
            "target": TargetContext(name=""),
            "auto": AUTO,
        }
        result.rendered = self._walk(
            mapping, context, resource=resource, path=path, phase="parse"
        )
        return result

    # -- resolve phase -----------------------------------------------------

    def resolve(
        self,
        mapping: dict[str, Any],
        ctx: ResolveContext,
        *,
        resource: str,
        path: Path,
    ) -> dict[str, Any]:
        """Fully render a spec mapping against the selected target."""

        def env_var(name: str, default: str | None = None) -> str:
            value = os.environ.get(name)
            if value is None:
                if default is None:
                    raise CompilationError(
                        f"environment variable {name!r} is not set",
                        resource=resource,
                        hint=f"export {name}=... or provide a default",
                    )
                return default
            return taint(value)

        context = {
            "ref": ctx.ref_resolver,
            "source": ctx.source_resolver,
            "var": ctx.lookup_var,
            "env_var": env_var,
            "target": ctx.target,
            "auto": AUTO,
        }
        rendered = self._walk(mapping, context, resource=resource, path=path, phase="compile")
        assert isinstance(rendered, dict)
        return rendered

    # -- shared walking ----------------------------------------------------

    def _walk(
        self,
        value: Any,
        context: dict[str, Any],
        *,
        resource: str,
        path: Path,
        phase: str,
    ) -> Any:
        if isinstance(value, dict):
            return {
                k: self._walk(v, context, resource=resource, path=path, phase=phase)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                self._walk(v, context, resource=resource, path=path, phase=phase)
                for v in value
            ]
        if isinstance(value, str):
            template_text = _normalize_template(value)
            if template_text is None:
                return value
            return self._render(template_text, context, resource=resource, path=path, phase=phase)
        return value

    def _render(
        self,
        template_text: str,
        context: dict[str, Any],
        *,
        resource: str,
        path: Path,
        phase: str,
    ) -> Any:
        error_cls = ConfigError if phase == "parse" else CompilationError
        try:
            return self._env.from_string(template_text).render(**context)
        except (ConfigError, CompilationError):
            raise
        except jinja2.UndefinedError as exc:
            raise error_cls(
                f"undefined Jinja name in {template_text!r}: {exc.message}",
                resource=resource,
                path=path,
                hint="available: ref, source, var, env_var, target, auto, and project macros",
            ) from exc
        except jinja2.TemplateError as exc:
            raise error_cls(
                f"invalid Jinja in {template_text!r}: {exc}",
                resource=resource,
                path=path,
            ) from exc
