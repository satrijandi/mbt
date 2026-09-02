#!/usr/bin/env python
"""Generate ``CHANGELOG.md`` from git tags and commit subjects.

    python scripts/generate_changelog.py            # rewrite CHANGELOG.md
    python scripts/generate_changelog.py --check    # fail if it is out of date

Ten packages ship from this repo and there was nowhere for a consumer to read
what changed between two of them (FEEDBACK v3 D-3). ``release.yml`` sets
``generate_release_notes: true``, which derives notes from merged pull
requests - and this repo has none, so a release's notes came out empty.

Generated rather than hand-written, deliberately: CONTRIBUTING said a manual
changelog "belongs to the deferred release pipeline, not a manual edit", and
that is still the right call. The source of truth is git, so the file cannot
drift from what actually shipped, and ``--check`` in CI keeps it honest the
same way ``test_version_sync.py`` keeps the version strings honest.

## Retraining impact

Every release section carries a **Retraining impact** line. This is not
boilerplate: mbt hashes the full spec dump, so a release that merely *adds* a
spec field changes every config hash and signals a full retrain on the next
``state:modified`` build (ADR-7). That is the single most expensive thing an
upgrade can do to a user, and it is invisible from a commit list, so it is
stated per release and defaults to the conservative answer.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

HEADER = """# Changelog

All notable changes to the mbt packages, newest first.
Generated from git history by `scripts/generate_changelog.py` - do not edit by
hand; run the script instead (CI checks it with `--check`).

Every release states its **Retraining impact**, because mbt hashes the whole
spec dump: a release that adds a spec field flips every config hash, so the
next `state:modified` build retrains everything (ADR-7). Read that line before
upgrading a project with expensive models.

"""

#: Commit-subject prefixes that are noise in a consumer-facing changelog.
_SKIP = re.compile(r"^(Merge |Bump version |wip\b)", re.IGNORECASE)

#: A conservative default: assume a release can flip config hashes unless a
#: human has recorded otherwise in RETRAINING_IMPACT below.
_DEFAULT_IMPACT = (
    "Not yet assessed - assume a full retrain on the next `state:modified` "
    "build (ADR-7) until a maintainer records otherwise here."
)

#: tag -> what upgrading does to config hashes. Maintainer-curated; the one
#: judgement in this file that git cannot answer.
RETRAINING_IMPACT: dict[str, str] = {
    "v0.1.0": "None - first release, so there is no prior manifest to diff against.",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def released_tags() -> list[str]:
    """Version tags, newest first, ordered by version not by string."""
    out = _git("tag", "--list", "v*", "--sort=-v:refname")
    return [line for line in out.splitlines() if line]


def subjects_between(older: str | None, newer: str) -> list[str]:
    """Commit subjects in ``(older, newer]``, newest first, noise dropped."""
    span = f"{older}..{newer}" if older else newer
    out = _git("log", "--no-merges", "--format=%s", span)
    return [line for line in out.splitlines() if line and not _SKIP.match(line)]


def _section(title: str, date: str | None, subjects: list[str], impact: str) -> str:
    heading = f"## {title}" + (f" - {date}" if date else "")
    lines = [heading, "", f"**Retraining impact:** {impact}", ""]
    if subjects:
        lines += [f"- {s}" for s in subjects]
    else:
        lines.append("- No user-visible changes.")
    lines.append("")
    return "\n".join(lines)


def render() -> str:
    tags = released_tags()
    parts = [HEADER]

    unreleased = subjects_between(tags[0], "HEAD") if tags else []
    if unreleased:
        parts.append(_section("Unreleased", None, unreleased, _DEFAULT_IMPACT))

    for index, tag in enumerate(tags):
        previous = tags[index + 1] if index + 1 < len(tags) else None
        date = _git("log", "-1", "--format=%ad", "--date=short", tag)
        impact = RETRAINING_IMPACT.get(tag, _DEFAULT_IMPACT)
        parts.append(_section(tag, date, subjects_between(previous, tag), impact))

    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if CHANGELOG.md differs from what git says.",
    )
    args = parser.parse_args(argv)

    rendered = render()
    if not args.check:
        CHANGELOG.write_text(rendered)
        print(f"wrote {CHANGELOG}")
        return 0

    current = CHANGELOG.read_text() if CHANGELOG.is_file() else ""
    if current == rendered:
        print("CHANGELOG.md is up to date")
        return 0
    print(
        "CHANGELOG.md is out of date. Regenerate it:\n    python scripts/generate_changelog.py",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
