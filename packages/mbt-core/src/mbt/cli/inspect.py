"""The read-only inspection commands' real work, off the typer surface (A-2).

``mbt ls`` and ``mbt show`` are not typer plumbing: between them they hold a
node view, four output formats, a resource lookup with a did-you-mean, and the
secret redaction that keeps a rendered ``env_var()`` out of a printed spec.
None of it was reachable except through ``CliRunner.invoke``, which is what
made ``main.py`` "a module pretending to be a CLI adapter" - 32 of the 42 tests
in ``test_cli_main_unit.py`` drove the runner, and the composition root itself
was never called directly by any test in the repo.

So the work lives here as ``*_cmd(cli, ...)`` functions that take an already
composed :class:`CLIContext` and RETURN their output instead of printing it.
``main.py`` keeps the option declarations and does the printing.

Lazy imports are load-bearing here as everywhere (ADR-14): the ML frameworks and
the compiler must not be imported by ``import mbt.cli.main``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mbt.cli.common import CLIContext


@dataclass(frozen=True)
class ListedResource:
    """One row of ``mbt ls``, independent of how it is rendered."""

    unique_id: str
    name: str
    resource_type: str
    tags: tuple[str, ...]
    path: str


def ls_cmd(
    cli: "CLIContext",
    *,
    select: list[str] | None = None,
    exclude: list[str] | None = None,
    state: str | None = None,
    state_include_env: bool = False,
    anchor: str | None = None,
) -> list[ListedResource]:
    """The resources ``mbt ls`` would list, in unique_id order.

    Selection runs over ``Manifest.selectable_nodes()`` with the same state
    index ``mbt build`` uses, so "the same selectors --select accepts" is true
    by construction rather than by two node views agreeing (v5 live defect 3:
    ``ls`` hand-rolled its view off the parsed project and never passed
    ``state``, so ``--select state:modified`` could not work and ``--state`` was
    not even an option).
    """
    from mbt.dag.selector import select_nodes
    from mbt.state.diff import ManifestStateIndex, load_state

    parsed, manifest = cli.compile(anchor=anchor)
    nodes = manifest.selectable_nodes()
    paths: dict[str, str] = {uid: node.path for uid, node in manifest.nodes.items()}
    paths.update({uid: source.path for uid, source in manifest.sources.items()})
    paths.update({uid: exposure.path for uid, exposure in manifest.exposures.items()})

    index = None
    if state is not None:
        reference = load_state(cli.resolve_cli_path(state) or state)
        index = ManifestStateIndex(manifest, reference, include_env=state_include_env)

    selected = sorted(select_nodes(parsed.graph, nodes, select, exclude, state=index))
    return [
        ListedResource(
            unique_id=uid,
            name=nodes[uid].name,
            resource_type=nodes[uid].resource_type,
            tags=tuple(nodes[uid].tags),
            path=paths.get(uid, ""),
        )
        for uid in selected
    ]


def show_cmd(
    cli: "CLIContext", name: str, *, output: str = "yaml", anchor: str | None = None
) -> str:
    """One resource's compile-rendered config, as the text ``mbt show`` prints.

    Redacted: a spec field may render an ``env_var()`` value (the jinja resolve
    context taints it), so this echoes rendered config the same way the manifest
    file does, through ``redact`` (NFR-07).
    """
    import json

    import yaml

    from mbt.exceptions import ConfigError
    from mbt.secrets import redact
    from mbt.utils import did_you_mean

    parsed, manifest = cli.compile(anchor=anchor)
    found: dict[str, Any] | None = None
    for pool in (manifest.nodes, manifest.sources, manifest.exposures):
        for uid, resource in pool.items():
            if name in (uid, resource.name):
                found = resource.model_dump(mode="json")
                break
        if found:
            break
    if found is None:
        suggestion = did_you_mean(name, parsed.all_names())
        raise ConfigError(
            f"unknown resource {name!r}",
            hint=f"did you mean {suggestion!r}?" if suggestion else "run 'mbt ls'",
        )
    if output == "json":
        return redact(json.dumps(found, indent=2))
    return redact(yaml.safe_dump(found, sort_keys=False, default_flow_style=False))


def clean_target_cmd(cli: "CLIContext") -> list[str]:
    """Delete ``target/`` and age out leaked job-payload dirs; return the notes.

    Goes through ``cli.project_dir``, which is resolved and chdir'd. The command
    body used to call ``shutil.rmtree`` on the raw typer path with no context at
    all (A-2), so ``--project-dir`` was interpreted relative to the invocation
    cwd here and relative to nothing anywhere else.
    """
    import shutil
    from datetime import UTC, datetime, timedelta

    from mbt.adapters.local.compute import sweep_stale_job_payloads

    notes: list[str] = []
    target_dir = cli.project_dir / "target"
    if target_dir.is_dir():
        shutil.rmtree(target_dir)
        notes.append(f"removed {target_dir}")
    else:
        notes.append(f"nothing to clean at {target_dir}")
    # Age out leaked error-payload dirs (kept for debugging, never
    # self-cleaned); recent ones survive for an in-progress reproduction.
    swept = sweep_stale_job_payloads(datetime.now(tz=UTC) - timedelta(days=7))
    if swept:
        notes.append(f"aged out {len(swept)} stale job payload dir(s) (>7d old)")
    return notes


__all__ = ["ListedResource", "clean_target_cmd", "ls_cmd", "show_cmd"]
