"""Print ``name==version`` for every package in a uv.lock, sorted.

Diffing two of these is how the upstream-resolution tier says what moved. A raw
``git diff uv.lock`` is close to unreadable for that question: one version bump
rewrites its wheel URLs and hashes, so a handful of real changes arrive as
hundreds of lines, and this week a breaking h2o upgrade hid inside exactly that
noise.

Stdlib only, so it runs before the workspace is synced (and therefore before
whatever broke the resolution can stop it from running).
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def versions(lock_text: str) -> list[str]:
    """``name==version`` for every locked package, sorted for a stable diff."""
    data = tomllib.loads(lock_text)
    return sorted(f"{pkg['name']}=={pkg['version']}" for pkg in data.get("package", []))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(f"usage: {Path(__file__).name} <path-to-uv.lock>", file=sys.stderr)
        return 2
    print("\n".join(versions(Path(args[0]).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
