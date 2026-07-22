#!/usr/bin/env python
"""Bump the mbt version in lockstep across every packaged version string.

One release edits 21 strings - the root plus ten package ``pyproject.toml``
``version`` fields and each package's runtime ``__version__`` - which
``tests/test_version_sync.py`` requires to agree. Doing that by hand is the
"monorepo release tax"; this does it in one command:

    python scripts/bump_version.py 0.2.0

It replaces the exact current-version string (read from the root
``pyproject.toml``) and fails loudly if any file does not carry it exactly
once, so a stray dependency pin is never rewritten by accident. See
CONTRIBUTING's "Releasing" section for the surrounding procedure (commit, then
tag ``vX.Y.Z``).
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def _replace_once(path: Path, old: str, new: str) -> Path:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{path}: expected exactly one {old!r}, found {count}")
    path.write_text(text.replace(old, new, 1))
    return path


def bump_version(root: Path, new_version: str) -> list[Path]:
    """Rewrite every packaged version string to ``new_version`` in lockstep.

    Returns the files changed (root pyproject + each package pyproject + each
    package ``__init__``). Raises ``ValueError`` on a malformed version, a
    no-op bump, or any file that does not carry the current version exactly once.
    """
    if not VERSION_RE.fullmatch(new_version):
        raise ValueError(f"expected an X.Y.Z version, got {new_version!r}")
    root_pyproject = root / "pyproject.toml"
    current = tomllib.loads(root_pyproject.read_text())["project"]["version"]
    if current == new_version:
        raise ValueError(f"version is already {new_version}")

    package_pyprojects = sorted(root.glob("packages/*/pyproject.toml"))
    changed: list[Path] = [
        _replace_once(pyproject, f'version = "{current}"', f'version = "{new_version}"')
        for pyproject in [root_pyproject, *package_pyprojects]
    ]
    for pyproject in package_pyprojects:
        (init,) = (pyproject.parent / "src").glob("*/__init__.py")
        changed.append(
            _replace_once(init, f'__version__ = "{current}"', f'__version__ = "{new_version}"')
        )
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="the new version, e.g. 0.2.0")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="workspace root (default: the repo containing this script)",
    )
    args = parser.parse_args(argv)
    try:
        changed = bump_version(args.root, args.version)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"bumped to {args.version} across {len(changed)} files:")
    for path in changed:
        print(f"  {path.relative_to(args.root)}")
    print("\nnext: review the diff, run the suite, commit, then tag vX.Y.Z (see CONTRIBUTING).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
