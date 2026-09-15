"""Guard: the spec examples in the documentation must be specs mbt accepts.

The first YAML a reader sees on the docs landing page failed `mbt parse`
(`evaluation.protocol` is required and it had none), the mbt-h2o README's model
had no `owner`, and the mbt-snowflake README still declared a dataset in the
`inputs:` form ADR-29 removed. Nothing read those blocks, so each one taught a
spec that could not run.

This validates every fenced YAML block that declares `models`, `datasets`,
`scoring`, or `sources` against the same Pydantic models the parser uses. It
does not resolve refs or render Jinja: an example is a snippet, not a project.
A block that is deliberately partial says so with a `...` line (or a
`# ...` comment) and is skipped; every other block must be valid YAML and a
valid spec, so a broken example cannot pass as a fragment. ADRs are excluded
on purpose - superseded ones describe removed syntax as the historical record.
"""

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mbt_adapter_base.specs import DatasetSpec, ModelSpec, ScoringSpec, SourceGroup

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC_MODELS = {
    "models": ModelSpec,
    "datasets": DatasetSpec,
    "scoring": ScoringSpec,
    "sources": SourceGroup,
}
_FENCE = re.compile(r"^```ya?ml\n(.*?)^```", re.MULTILINE | re.DOTALL)
_ELISION = re.compile(r"^\s*(#\s*)?\.\.\.", re.MULTILINE)


def _documents() -> list[Path]:
    docs = [p for p in (REPO_ROOT / "docs").rglob("*.md") if "adr" not in p.parts]
    readmes = [
        REPO_ROOT / "README.md",
        *(REPO_ROOT / "packages").glob("*/README.md"),
        *(REPO_ROOT / "tests" / "fixtures").glob("*/README.md"),
        REPO_ROOT / "examples" / "showcase" / "README.md",
    ]
    return sorted(docs + readmes)


def _blocks() -> list[tuple[str, str]]:
    found = []
    for path in _documents():
        text = path.read_text()
        for match in _FENCE.finditer(text):
            line = text[: match.start()].count("\n") + 1
            found.append((f"{path.relative_to(REPO_ROOT)}:{line}", match.group(1)))
    return found


BLOCKS = _blocks()


def test_the_scan_finds_the_examples_it_exists_for() -> None:
    """If the fence pattern stops matching, every other test here passes vacuously."""
    spec_blocks = [where for where, body in BLOCKS if re.search(r"^(models|datasets):", body, re.M)]
    assert len(spec_blocks) >= 8, spec_blocks
    assert any(where.startswith("docs/index.md") for where in spec_blocks)
    assert any(where.startswith("README.md") for where in spec_blocks)


@pytest.mark.parametrize(("where", "body"), BLOCKS, ids=[where for where, _ in BLOCKS])
def test_yaml_example_is_a_valid_spec(where: str, body: str) -> None:
    if _ELISION.search(body):
        return  # declared partial: a snippet that elides required fields on purpose
    try:
        document = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        pytest.fail(f"{where}: not valid YAML, and not marked partial with a '...' line: {exc}")
    if not isinstance(document, dict):
        return
    for key, model in _SPEC_MODELS.items():
        entries = document.get(key)
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            assert isinstance(entry, dict), f"{where}: {key}[{index}] is not a mapping"
            try:
                model.model_validate(entry)
            except ValidationError as exc:
                problems = "; ".join(
                    f"/{'/'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
                )
                pytest.fail(f"{where}: {key}[{index}] is not a valid {model.__name__}: {problems}")
