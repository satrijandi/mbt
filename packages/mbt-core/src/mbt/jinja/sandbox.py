"""The one place a Jinja environment is built (FEEDBACK v7 B-1).

``SECURITY.md`` defines "a spec, profile, or hook executes code outside the
documented seams" as a vulnerability, with Python hooks and adapters the only
code by design: YAML and Jinja are not. An unsandboxed environment breaks
that, because ``{{ lipsum.__globals__['os'].getpid() }}`` reaches the ``os``
module through any function the environment exposes. Spec rendering was
sandboxed from the start; ``profiles.yml`` and the job's re-render of adapter
config were not, and ``profiles.yml`` is committed and rendered on every pull
request's CI run.

So an environment is built here and nowhere else, and
``test_jinja_sandbox_guard.py`` fails on any direct construction of an
unsandboxed environment anywhere in ``packages/*/src``, this module included.
"""

import jinja2
from jinja2.nativetypes import NativeCodeGenerator, NativeTemplate, native_concat
from jinja2.sandbox import SandboxedEnvironment


class NativeSandbox(SandboxedEnvironment):
    """Sandboxed environment that preserves native Python types."""

    code_generator_class = NativeCodeGenerator
    template_class = NativeTemplate
    concat = staticmethod(native_concat)  # type: ignore[assignment]


def sandboxed_environment() -> SandboxedEnvironment:
    """A sandboxed environment that renders to ``str``, for text templates."""
    return SandboxedEnvironment(undefined=jinja2.StrictUndefined, autoescape=False)


def native_sandboxed_environment() -> NativeSandbox:
    """A sandboxed environment whose renders keep their Python type, for specs."""
    return NativeSandbox(undefined=jinja2.StrictUndefined, autoescape=False)
