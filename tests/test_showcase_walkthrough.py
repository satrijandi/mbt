"""Hermetic guard for the showcase walkthrough (docs/showcase-walkthrough.md).

The walkthrough is a page a user types from, so every fact it hard-codes -
accounts, passwords, host ports, pinned anchors, make targets - must match
the stack it describes. The live tiers prove the stack; this proves the page
still describes that stack, without docker.
"""

import ast
import re
from pathlib import Path

from showcase_utils import ANCHOR, MONITOR_ANCHOR

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "showcase"
PAGE = REPO_ROOT / "docs" / "showcase-walkthrough.md"


def _text() -> str:
    return PAGE.read_text()


def _bootstrap_constants() -> dict[str, str]:
    tree = ast.parse((SHOWCASE / "scripts" / "ci_bootstrap.py").read_text())
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _default_ports() -> set[str]:
    env = (SHOWCASE / ".env.example").read_text()
    return set(re.findall(r"^#?SHOWCASE_\w+_PORT=(\d+)$", env, re.MULTILINE))


def test_accounts_and_passwords_match_the_bootstrap() -> None:
    text, consts = _text(), _bootstrap_constants()
    for key in ("USER", "PASSWORD", "DS_USER", "DS_PASSWORD"):
        assert consts[key] in text, key
    assert f"{consts['ORG']}/{consts['REPO']}" in text
    assert f"{consts['ORG']}/{consts['DEPLOY_REPO']}" in text


def test_every_localhost_port_is_a_published_default() -> None:
    used = set(re.findall(r"localhost:(\d+)", _text()))
    assert used, "no localhost URLs found"
    assert used <= _default_ports(), sorted(used - _default_ports())


def test_the_pinned_anchors_are_the_live_tiers() -> None:
    text = _text()
    assert ANCHOR in text and MONITOR_ANCHOR in text
    assert set(re.findall(r"\d{4}-\d\d-\d\dT00:00:00Z", text)) <= {ANCHOR, MONITOR_ANCHOR}


def test_every_make_target_exists() -> None:
    makefile = (SHOWCASE / "Makefile").read_text()
    defined = set(re.findall(r"^([a-z][a-z-]*):", makefile, re.MULTILINE))
    used = {target for line in re.findall(r"`make ([a-z -]+)`", _text()) for target in line.split()}
    used |= set(re.findall(r"^make ([a-z-]+)", _text(), re.MULTILINE))
    assert used, "no make targets found"
    assert used <= defined, sorted(used - defined)
