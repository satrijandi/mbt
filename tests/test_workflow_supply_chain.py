"""Supply-chain guards for every workflow this repo runs AND every workflow it
ships (FEEDBACK A-2, B-2, C-1).

Three properties, asserted over the repo's `.github/workflows` and the
scaffold's, because the scaffold is copied into user projects and propagates
whatever it models:

1. Every third-party action is pinned to a 40-hex commit SHA.
   `renovate.json` has extended `helpers:pinGitHubActionDigests` since it was
   written, and none of the fourteen actions were pinned - a stated protection
   with no control behind it, which reads to the next person as "reviewed".
   `release/v1` on `pypa/gh-action-pypi-publish` was the sharpest case: a
   *branch*, the most mutable ref there is, holding the PyPI publish credential.

2. Every workflow declares a top-level `permissions:` block.
   Without one the jobs inherit the repository default, which is exactly the
   set of jobs that run `pip install`, `uv sync`, and third-party actions.

3. No `${{ }}` expression is spliced into a `run:` script.
   Template substitution happens before bash parses the line, so an input can
   inject arguments. `promote.yml` rendered `--version ${{ inputs.version }}`
   unquoted, where `1 --force` reaches `mbt promote --version 1 --force` -
   `--force` being the documented override for "gates were not recorded as
   passed", on the workflow whose purpose is enforcing that gate.
"""

import re
from pathlib import Path

import pytest
import yaml

import mbt

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SCAFFOLD_WORKFLOWS = (
    Path(mbt.__file__).resolve().parent / "cli" / "_scaffold" / ".github" / "workflows"
)

WORKFLOWS = sorted(REPO_WORKFLOWS.glob("*.yml")) + sorted(SCAFFOLD_WORKFLOWS.glob("*.yml"))

#: `uses: owner/repo[/path]@ref`, with any trailing comment.
_USES = re.compile(
    r"^\s*(?:-\s+)?uses:\s+(?P<action>[^\s@#]+)@(?P<ref>[^\s#]+)(?:\s*#\s*(?P<comment>.*))?$"
)
_SHA = re.compile(r"^[0-9a-f]{40}$")
_EXPRESSION = re.compile(r"\$\{\{")


def _ids(paths: list[Path]) -> list[str]:
    return [f"{p.parent.parent.parent.name}/{p.name}" for p in paths]


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=_ids(WORKFLOWS))
def test_actions_are_pinned_to_commit_digests(workflow: Path) -> None:
    for line in workflow.read_text().splitlines():
        match = _USES.match(line)
        if not match:
            continue
        action, ref = match.group("action"), match.group("ref")
        if action.startswith("./"):
            continue  # a local reusable workflow is this repo's own tree
        assert _SHA.match(ref), (
            f"{workflow.name}: {action}@{ref} is a mutable ref. Pin it to a commit "
            "SHA with a trailing '# vX.Y.Z' comment (renovate.json extends "
            "helpers:pinGitHubActionDigests, which maintains both once pinned)."
        )
        # The comment is the human-readable half: without it nobody can tell
        # what version a 40-hex string is, and the pin stops getting reviewed.
        assert (match.group("comment") or "").startswith("v"), (
            f"{workflow.name}: {action} is pinned but carries no '# vX.Y.Z' version comment"
        )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=_ids(WORKFLOWS))
def test_workflows_declare_least_privilege_permissions(workflow: Path) -> None:
    spec = yaml.safe_load(workflow.read_text())
    permissions = spec.get("permissions")
    assert isinstance(permissions, dict) and permissions, (
        f"{workflow.name} declares no top-level permissions: its jobs inherit "
        "the repository default"
    )
    assert permissions.get("contents") == "read" or workflow.name == "release.yml", (
        f"{workflow.name} grants {permissions.get('contents')!r} at workflow scope; "
        "keep contents: read there and let the one job that writes opt up"
    )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=_ids(WORKFLOWS))
def test_no_template_expressions_inside_run_scripts(workflow: Path) -> None:
    spec = yaml.safe_load(workflow.read_text())
    for job_name, job in (spec.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            script = step.get("run")
            if not script:
                continue
            offending = [line for line in script.splitlines() if _EXPRESSION.search(line)]
            assert not offending, (
                f"{workflow.name}:{job_name} splices a template expression into a "
                f"run script: {offending}. Pass the value through env: and "
                'reference it as "$NAME" so bash never parses attacker-chosen text.'
            )
