"""The CLI must keep catching typer/click control-flow exceptions, wherever
upstream has moved them this month.

typer >= 0.20 vendors click, and its commands raise the VENDORED exception
types, which are not subclasses of the real click's - so `mbt` has to catch
both or it exits on its own control flow (a `typer.Exit(0)` reaching the
generic handler prints "Internal error" and exits 1 on a successful command).

Which module holds which name is not stable. typer 0.27.2 moved `Exit` and
`Abort` out of `typer._click.exceptions` into `typer.exceptions` as plain
RuntimeErrors and left `UsageError`/`ClickException` behind. The old code
guarded the *import*, which that move sails through - the module still
imports, it has just lost two attributes - so every `mbt` invocation died
while evaluating an except clause:

    except (click.exceptions.Exit, typer_click_exc.Exit) as exc:
                                   ^^^^^^^^^^^^^^^^^^^^
    AttributeError: module 'typer._click.exceptions' has no attribute 'Exit'

Not a test failure - the CLI, gone, for anyone installing unpinned. The
nightly upstream-resolution tier caught it; uv.lock hid it from every other
check because the locked typer was 0.27.1.

`_control_flow_exceptions` now probes by name across every module that has
ever held these, and returns an empty tuple for a name found nowhere, so a
FUTURE move degrades to "we stopped special-casing that exception" instead of
crashing. That degradation is silent by construction, which is what these
tests are for.
"""

import click
import pytest
import typer

from mbt.cli import main as cli_main

NAMES = ["Exit", "UsageError", "ClickException", "Abort"]


@pytest.mark.parametrize("name", NAMES)
def test_every_control_flow_exception_resolves_somewhere(name: str) -> None:
    """An empty tuple never matches, so mbt would silently stop honouring
    that exception - a `typer.Exit(2)` would surface as "Internal error"."""
    resolved = cli_main._control_flow_exceptions(name)
    assert resolved, (
        f"no {name} found in click.exceptions, click, typer, or typer._click."
        f"exceptions - upstream moved it again; add the new home to "
        f"_control_flow_exceptions' source list"
    )


def test_the_public_typer_exceptions_are_caught() -> None:
    """`typer.Exit` / `typer.Abort` are typer's stable public names and what
    mbt's own commands raise; whatever the vendoring looks like, these two must
    be in the tuples."""
    assert typer.Exit in cli_main.EXIT_EXCEPTIONS
    assert typer.Abort in cli_main.ABORT_EXCEPTIONS


def test_the_real_click_exceptions_are_caught() -> None:
    """Not redundant with the typer ones: under a vendoring typer these are
    genuinely different classes with no subclass relationship, and plain click
    decorators are still reachable through the app."""
    assert click.exceptions.Exit in cli_main.EXIT_EXCEPTIONS
    assert click.UsageError in cli_main.USAGE_ERROR_EXCEPTIONS
    assert click.ClickException in cli_main.CLICK_EXCEPTIONS
    assert click.Abort in cli_main.ABORT_EXCEPTIONS


def test_resolution_survives_a_module_that_lost_the_attribute() -> None:
    """The 0.27.2 shape exactly: the module imports, the attribute is gone.

    Resolving must not raise - that is the whole regression - and must still
    return the classes the other sources provide.
    """
    resolved = cli_main._control_flow_exceptions("NoSuchExceptionAnywhere")
    assert resolved == ()

    # And a name only some sources carry still resolves from the others.
    assert cli_main._control_flow_exceptions("Exit")


def test_only_exception_classes_are_collected() -> None:
    """`getattr` by name across four modules will find non-exceptions sooner
    or later (typer._click.exceptions already exports `echo` and `Any`);
    putting a non-class in an except clause is a TypeError at raise time."""
    for name in NAMES:
        for candidate in cli_main._control_flow_exceptions(name):
            assert isinstance(candidate, type)
            assert issubclass(candidate, BaseException)


def test_no_duplicates_so_except_clauses_stay_readable() -> None:
    """Under typer < 0.20 every source is the real click, so the naive version
    of this would build a tuple of the same class four times."""
    for name in NAMES:
        resolved = cli_main._control_flow_exceptions(name)
        assert len(resolved) == len(set(resolved)), (name, resolved)
