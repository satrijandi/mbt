"""Guard: the prose that states the experiment-naming rule must match the code.

The rule - `<project>__<experiment>` - is composed in exactly one place
(`mbt.runtime.tracking_adapter_config`) and then restated in prose in a dozen
files: the spec reference, ADR-28, the scaffold template a user's project is
stamped from, and the showcase's README/DESIGN/Makefile. Those restatements
have no other test. ADR-28's own separator sat wrong in four showcase files for
the length of a release once the rule moved, and nothing caught it.

`examples/showcase/**` is the sharp end of this: the live_showcase tier is
gated, so `-m e2e` excludes it and `-m "not e2e"` deselects it, and README /
DESIGN / Makefile are read by no test at all. These two run in the fast tier.
"""

import re
from pathlib import Path

import yaml

from mbt.runtime import EXPERIMENT_SEPARATOR

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "showcase"

#: Anchored on the literal `<experiment>` so docker compose's `<project>_default`
#: network naming (compose/docker-compose.yml, showcase_utils.py) cannot match.
_RULE_RE = re.compile(r"<project>(_+)<experiment>")

#: Everything a reader could take the rule from. design-history/ is excluded on
#: purpose: it is pre-implementation drafting the ADRs supersede, and rewriting
#: it would be rewriting history.
_SEARCH_ROOTS = (
    REPO_ROOT / "docs",
    REPO_ROOT / "packages" / "mbt-core" / "src" / "mbt",
    REPO_ROOT / "tests",
    SHOWCASE,
)
_SUFFIXES = {".md", ".py", ".yml", ".yaml"}


def _candidate_files():
    for root in _SEARCH_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix in _SUFFIXES and "__pycache__" not in path.parts:
                yield path


def test_every_statement_of_the_rule_uses_the_separator_the_code_composes_with() -> None:
    wrong: list[str] = []
    seen = 0
    for path in _candidate_files():
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            for match in _RULE_RE.finditer(line):
                seen += 1
                if match.group(1) != EXPERIMENT_SEPARATOR:
                    wrong.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")

    assert seen, "no file states the experiment-naming rule - did the wording change?"
    assert not wrong, (
        f"prose states the rule with a separator the code does not use "
        f"({EXPERIMENT_SEPARATOR!r}):\n" + "\n".join(wrong)
    )


def test_the_showcase_names_each_planes_experiment_the_way_core_composes_it() -> None:
    """Derived from the showcase's own config, not hardcoded here.

    The showcase documents its two named planes by experiment name in README,
    DESIGN and the Makefile echo lines. Those are the strings a reader greps
    MLflow for, so a stale one sends them to an experiment that does not exist.
    """
    project = yaml.safe_load((SHOWCASE / "project" / "mbt_project.yml").read_text())["name"]
    outputs = yaml.safe_load((SHOWCASE / "project" / "profiles.yml").read_text())[project][
        "outputs"
    ]

    prose = "\n".join(
        (SHOWCASE / name).read_text() for name in ("README.md", "DESIGN.md", "Makefile")
    )
    checked = 0
    for target, config in outputs.items():
        declared = config.get("tracking", {}).get("config", {}).get("experiment")
        if not declared:
            continue  # lands in the bare project name; nothing composed to check
        checked += 1
        composed = f"{project}{EXPERIMENT_SEPARATOR}{declared}"
        assert composed in prose, (
            f"target {target!r} composes to {composed!r}, which the showcase prose never names"
        )
        stale = f"{project}_{declared}"
        assert stale not in prose, f"showcase prose still names {stale}, the single-underscore form"

    assert checked >= 2, f"expected the two named planes to set experiment:, found {checked}"
