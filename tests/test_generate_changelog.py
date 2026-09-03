"""The generated changelog (FEEDBACK v3 D-3).

Ten packages ship from this repo with `generate_release_notes: true` deriving
notes from merged pull requests - of which this repo has none, so a release's
notes came out empty. The changelog is generated from git instead, which is why
these tests are about the *generator's* invariants rather than the file's text:
the file moves with every commit, the rules do not.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generate_changelog as gen

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The changelog is generated FROM git, so these guards have no source of truth
#: other than the real repository. A tagless clone makes them fail in ways that
#: read as generator bugs (IndexError on `tags[0]`) rather than as a checkout
#: that fetched no tags, which is what `actions/checkout` does by default.
_NEEDS_TAGS = (
    "no v* tags in this clone - the changelog guards read real git history. "
    "CI checks out with fetch-depth: 0 for exactly this reason; locally, "
    "run `git fetch --tags`."
)


def _tags() -> list[str]:
    tags = gen.released_tags()
    assert tags, _NEEDS_TAGS
    return tags


def test_every_released_tag_gets_a_section() -> None:
    """A shipped version with no entry is the failure mode being fixed."""
    rendered = gen.render()
    tags = _tags()
    for tag in tags:
        assert f"## {tag} - " in rendered, f"{tag} missing from the changelog"


def test_every_section_states_its_retraining_impact() -> None:
    """The one fact git cannot derive, and the most expensive one to omit.

    A release that adds a spec field flips every config hash under full-dump
    hashing (ADR-7), so the next `state:modified` build retrains everything.
    A changelog that lists commits but not that is worse than none.
    """
    _tags()
    rendered = gen.render()
    headings = [line for line in rendered.splitlines() if line.startswith("## ")]
    assert headings
    assert rendered.count("**Retraining impact:**") == len(headings)


def test_curated_impacts_reference_real_tags() -> None:
    """RETRAINING_IMPACT is hand-maintained, so it can rot; a key that is not a
    tag means someone recorded impact for a release that does not exist."""
    assert set(gen.RETRAINING_IMPACT) <= set(_tags())


def test_merge_and_version_bump_commits_are_dropped() -> None:
    assert gen._SKIP.match("Merge pull request #3 from x")
    assert gen._SKIP.match("Bump version to 0.2.0")
    assert not gen._SKIP.match("Fix the scoring window resolution")


def test_check_mode_passes_on_a_freshly_generated_file(tmp_path: Path) -> None:
    """--check must agree with what the generator just wrote, or the release
    job is unpassable."""
    target = tmp_path / "CHANGELOG.md"
    target.write_text(gen.render())
    original = gen.CHANGELOG
    try:
        gen.CHANGELOG = target
        assert gen.main(["--check"]) == 0
    finally:
        gen.CHANGELOG = original


def test_check_mode_fails_when_the_file_drifts(tmp_path: Path, capsys) -> None:
    target = tmp_path / "CHANGELOG.md"
    target.write_text("# Changelog\n\nhand-edited\n")
    original = gen.CHANGELOG
    try:
        gen.CHANGELOG = target
        assert gen.main(["--check"]) == 1
    finally:
        gen.CHANGELOG = original
    assert "out of date" in capsys.readouterr().err


def test_a_missing_file_is_out_of_date_not_a_crash(tmp_path: Path) -> None:
    original = gen.CHANGELOG
    try:
        gen.CHANGELOG = tmp_path / "nope.md"
        assert gen.main(["--check"]) == 1
    finally:
        gen.CHANGELOG = original


def test_writing_produces_the_rendered_document(tmp_path: Path) -> None:
    original = gen.CHANGELOG
    try:
        gen.CHANGELOG = tmp_path / "out.md"
        assert gen.main([]) == 0
        assert (tmp_path / "out.md").read_text() == gen.render()
    finally:
        gen.CHANGELOG = original


def test_the_committed_changelog_is_in_sync_for_released_tags() -> None:
    """The "Unreleased" section moves with every commit, so the committed file
    is only pinned from the newest tag downwards - which is the part consumers
    read and the part `release.yml --check` enforces at tag time."""
    committed = (REPO_ROOT / "CHANGELOG.md").read_text()
    rendered = gen.render()
    newest = _tags()[0]
    assert (
        committed[committed.index(f"## {newest} - ") :]
        == (rendered[rendered.index(f"## {newest} - ") :])
    )


def test_the_script_runs_as_a_subprocess() -> None:
    """It is invoked by the release workflow, not imported, so the entrypoint
    has to work standalone.

    Asserts that it *reaches a verdict*, not which verdict. `--check` compares
    the whole file, and the "Unreleased" section gains a line on every commit,
    so demanding exit 0 here fails the fast suite on the first commit after any
    regeneration - which is precisely why `release.yml` runs `--check` at tag
    time instead, where that section is empty and the comparison is stable.
    The committed file's stable part is pinned by
    `test_the_committed_changelog_is_in_sync_for_released_tags` above.
    """
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_changelog.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = proc.stdout + proc.stderr
    assert proc.returncode in (0, 1), report
    assert "CHANGELOG.md is up to date" in report or "CHANGELOG.md is out of date" in report


def test_every_job_running_this_suite_checks_out_tags() -> None:
    """The other half of `_NEEDS_TAGS`, pinned so it cannot regress silently.

    These guards read git, and `actions/checkout` fetches a depth-1 clone with
    no tags by default, so the whole module passed locally and failed in every
    CI job that ran it. A job that runs the fast suite must ask for history.

    Scans EVERY workflow, not just ci.yml. The first version of this test read
    ci.yml alone, so it went green while upstream.yml - which also runs the
    fast suite - stayed red for two nights, reporting the missing tags as
    "the newest versions our constraints allow no longer pass". A guard that
    covers one file of a repo-wide invariant is how that red got misattributed.
    """
    import yaml

    checked = []
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            runs_fast_suite = any(
                "pytest" in str(step.get("run", "")) and "not e2e" in str(step.get("run", ""))
                for step in steps
            )
            if not runs_fast_suite:
                continue
            where = f"{path.name}:{name}"
            checked.append(where)
            checkout = next(
                (s for s in steps if str(s.get("uses", "")).startswith("actions/checkout")), None
            )
            assert checkout is not None, f"{where} runs the fast suite with no checkout step"
            assert (checkout.get("with") or {}).get("fetch-depth") == 0, (
                f"{where} runs the fast suite but checks out without tags; "
                "tests/test_generate_changelog.py reads real git history"
            )

    # Non-vacuity: a rename or a restructure that stops matching any job would
    # otherwise turn this into a test that passes by finding nothing.
    assert len(checked) >= 3, f"expected to find the fast-suite jobs, found {checked}"


def test_subjects_between_reads_the_tag_span() -> None:
    tags = _tags()
    subjects = gen.subjects_between(None, tags[-1])
    assert subjects, "the first release should have commits"
    assert all(not gen._SKIP.match(s) for s in subjects)


@pytest.mark.parametrize("impact", ["", "x"])
def test_sections_render_with_or_without_commits(impact: str) -> None:
    empty = gen._section("v9.9.9", "2026-01-01", [], impact)
    assert "No user-visible changes." in empty
    filled = gen._section("v9.9.9", None, ["did a thing"], impact)
    assert "- did a thing" in filled
    assert " - " not in filled.splitlines()[0]  # no date -> no dash suffix
