"""Pre-merge lint for promotions.yml (DESIGN.md section 6): every entry must
pin `version:`.

An unpinned staging-to-production entry resolves at promote time, and
promotion vacates the staging alias (aliases are exclusive per version), so
replaying the same file exits 1 with "no version in stage staging" - by
design. Pinning makes promotions.yml a reviewable, replayable GitOps record;
this lint rejects the footgun before it merges.
"""

import sys
from pathlib import Path

import yaml


def main() -> int:
    path = Path("promotions.yml")
    if not path.exists():
        print("lint_promotions: no promotions.yml - nothing to lint")
        return 0

    doc = yaml.safe_load(path.read_text()) or {}
    entries = doc.get("promotions") or []
    if not isinstance(entries, list):
        print("lint_promotions: `promotions` must be a list", file=sys.stderr)
        return 1

    unpinned = [
        entry.get("model", f"entry #{i}")
        for i, entry in enumerate(entries, 1)
        if not isinstance(entry, dict) or entry.get("version") in (None, "")
    ]
    if unpinned:
        print(
            "lint_promotions: entries without a pinned `version:`: "
            + ", ".join(str(u) for u in unpinned)
            + "\nUnpinned promotions are not replayable (promotion vacates the "
            "staging alias); pin the exact version you reviewed.",
            file=sys.stderr,
        )
        return 1

    print(f"lint_promotions: {len(entries)} pinned promotion(s) - ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
