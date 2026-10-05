"""Guard: prose that claims an ADR count must match the actual ADR files.

The v0.1 status page markets the project's rigor and had drifted (it claimed
"21 ADRs" after ADR-22 landed). This pins every "<N> ADRs" claim across the
docs to ``len(docs/adr/NNNN-*.md)`` so the count can never silently rot again.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Numbered ADR files (0001-*.md ...); the index page is not an ADR.
_ADR_RE = re.compile(r"^\d{4}-.*\.md$")
#: Prose claims like "22 ADRs" AND the markdown-link form "22 [ADRs](...)".
#: The optional ``[`` is what a bare ``\s+ADRs`` missed, so a stale link-form
#: count silently passed CI (the very drift this guard claims to prevent).
_CLAIM_RE = re.compile(r"(\d+)\s+\[?ADRs\b")


def _actual_adr_count() -> int:
    adr_dir = REPO_ROOT / "docs" / "adr"
    return sum(1 for p in adr_dir.iterdir() if _ADR_RE.match(p.name))


def test_doc_adr_counts_match_actual_files() -> None:
    actual = _actual_adr_count()
    assert actual > 0, "no ADR files found - path wrong?"

    mismatches: list[str] = []
    for md in sorted((REPO_ROOT / "docs").rglob("*.md")):
        for line_no, line in enumerate(md.read_text().splitlines(), start=1):
            for claim in _CLAIM_RE.finditer(line):
                if int(claim.group(1)) != actual:
                    rel = md.relative_to(REPO_ROOT)
                    mismatches.append(
                        f"{rel}:{line_no} claims {claim.group(1)} ADRs, actual is {actual}"
                    )

    assert not mismatches, "stale ADR counts in docs:\n" + "\n".join(mismatches)


def test_adr_23_does_not_claim_a_shipped_capability_is_deferred() -> None:
    """ADRs are the repo's "read this before you 'fix' something odd" layer, so
    one that describes the code as it was rather than as it is costs more than
    no ADR at all.

    ADR-23 deferred Spark warehouse scoring to a follow-up; the follow-up
    shipped in 6c399c9 and the ADR was not updated for a month, so the only
    open issue on the repo cites it to claim both adapters lack methods they
    have. Assert the two agree: if a warehouse adapter implements contract 1.1,
    the ADR must not still be calling it deferred.
    """
    adr = (REPO_ROOT / "docs" / "adr" / "0023-warehouse-batch-scoring.md").read_text()

    implementations = {
        "Spark": REPO_ROOT / "packages" / "mbt-spark" / "src" / "mbt_spark" / "data.py",
        "Snowflake": REPO_ROOT
        / "packages"
        / "mbt-snowflake"
        / "src"
        / "mbt_snowflake"
        / "adapter.py",
    }
    for adapter, source in implementations.items():
        text = source.read_text()
        ships = "def build_scoring_input" in text and "def open_predictions" in text
        assert ships, f"{adapter} no longer implements contract 1.1; ADR-23 needs revisiting"

    # The section that went stale. "landed" is the correction; if someone
    # reverts the implementation they must revert this too.
    assert "**That follow-up landed**" in adr, (
        "ADR-23 still presents Spark warehouse scoring as deferred, but "
        "mbt_spark implements build_scoring_input and open_predictions"
    )


# -- the JVM coverage floor must stay wired in (FEEDBACK v3 B-3) ---------------


def test_the_jvm_coverage_floor_is_real_and_enforced() -> None:
    """`docs/v0.1-status.md` claims the e2e tier covers mbt-spark/mbt-h2o.

    That was an assertion nobody measured until 2026-09-01. It is now a
    measured floor, and this guards the two halves that could silently
    decouple: the config must declare a floor over exactly those packages, and
    the e2e job must actually pass the config.
    """
    import configparser

    config = configparser.ConfigParser()
    config.read(REPO_ROOT / "tests" / "coverage-jvm.cfg")
    packages = config["run"]["source_pkgs"].split()
    assert sorted(packages) == ["mbt_h2o", "mbt_spark"], packages
    assert int(config["report"]["fail_under"]) >= 80, "the floor must stay a real bar"

    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "--cov-config=tests/coverage-jvm.cfg" in workflow, (
        "the e2e job stopped passing the JVM coverage config, so the floor is inert"
    )


def test_the_jvm_packages_are_outside_the_fast_suite_gate() -> None:
    """The two configs must not overlap: mbt-spark/mbt-h2o in the fast suite's
    source_pkgs would make its 100% gate unreachable, which is why they are
    measured separately in the first place."""
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    fast = set(pyproject["tool"]["coverage"]["run"]["source_pkgs"])
    assert not fast & {"mbt_spark", "mbt_h2o"}, fast


# -- the status page's other counts (FEEDBACK v6 D-1) ---------------------------------

#: "12 publishable packages", "12 wheels", "all 12 packages" - and the spelled-
#: out forms the release workflow and changelog script used ("Ten packages").
_PACKAGE_CLAIM_RE = re.compile(
    r"\b(\d+|ten|eleven|twelve|thirteen)\s+(?:publishable\s+)?(packages|wheels/sdists|wheels)\b",
    re.IGNORECASE,
)
_WORDS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13}


def test_package_counts_match_the_workspace() -> None:
    """The status page said 11 publishable packages and release.yml said 10
    wheels while ``packages/`` held 12, and CLAUDE.md asks that the page stay
    exactly true. Every such claim is derived from the workspace now."""
    actual = len(list((REPO_ROOT / "packages").glob("*/pyproject.toml")))
    files = [
        REPO_ROOT / "docs" / "v0.1-status.md",
        REPO_ROOT / ".github" / "workflows" / "release.yml",
        REPO_ROOT / "scripts" / "generate_changelog.py",
        REPO_ROOT / "CONTRIBUTING.md",
    ]
    mismatches = []
    for path in files:
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            for claim in _PACKAGE_CLAIM_RE.finditer(line):
                number, noun = claim.group(1).lower(), claim.group(2).lower()
                stated = int(number) if number.isdigit() else _WORDS[number]
                expected = 2 * actual if noun == "wheels/sdists" else actual  # one of each
                if stated != expected:
                    mismatches.append(
                        f"{path.relative_to(REPO_ROOT)}:{n}: {claim.group(0)!r}, expected "
                        f"{expected}"
                    )
    assert not mismatches, "stale package counts:\n" + "\n".join(mismatches)


def test_the_status_page_states_no_exact_test_count() -> None:
    """ "1699 tests (1611 fast, 88 e2e)" was stale within weeks: every commit
    moves it, and no check could keep it true. The page states what the suite
    covers and how that is enforced, not a number."""
    page = (REPO_ROOT / "docs" / "v0.1-status.md").read_text()
    assert not re.search(r"\b\d[\d,]*\s+(?:fast\s+|e2e\s+)?tests\b", page)
