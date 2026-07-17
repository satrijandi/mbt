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
#: Prose claims like "22 ADRs".
_CLAIM_RE = re.compile(r"(\d+)\s+ADRs\b")


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
