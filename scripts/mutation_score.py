#!/usr/bin/env python
"""Mutation-test the code that decides whether a model ships, with a ratchet.

    uv run --group mutation python scripts/mutation_score.py
    uv run --group mutation python scripts/mutation_score.py --update-baseline

CI enforces 100% line coverage, which proves every line RAN, not that any test
would notice it being wrong. FEEDBACK v6 C-1 measured the difference by hand:
of 20 single-token mutants over gates, verdicts and monitors, 10 survived the
whole fast suite - one of them let a model whose stability monitors breached
register anyway. This makes that measurement repeatable. ``mutmut`` mutates
the modules in ``TARGETS``, runs the owning package's fast tests against each
mutant, and counts the survivors.

It is a ratchet, not a gate: the run fails when a target has MORE survivors
than ``scripts/mutation_baseline.json`` records, and asks for the baseline to
come down when it has fewer. A survivor is not automatically a missing test -
mutating an error message's wording survives by design - so the number is
read as a trend, and ``--update-baseline`` is for lowering it, with the diff
reviewed like any other.

Each package is mutated in a throwaway copy: mutmut assumes one package with
its source under ``src/`` and its tests beside it, which is what a single
workspace member is, and never what the repo root is.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "scripts" / "mutation_baseline.json"


@dataclass(frozen=True)
class Target:
    """One workspace package, and the modules in it worth mutating."""

    package: str
    only_mutate: tuple[str, ...]


#: The code that decides whether a model ships or a batch alarms (C-1): gates,
#: the verdict, monitors and checks; the seed ladder every seeded stage reads;
#: and the metrics and paired bootstrap the gates compare.
TARGETS = (
    Target("mbt-core", ("src/mbt/quality/*", "src/mbt/execute/seeds.py")),
    Target("mbt-adapter-base", ("src/mbt_adapter_base/metrics.py",)),
)

_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "mutants", ".pytest_cache", ".hypothesis", "*.pyc"
)


def _setup_cfg(target: Target) -> str:
    only = "\n    ".join(target.only_mutate)
    return (
        "[mutmut]\n"
        "source_paths=src\n"
        f"only_mutate={only}\n"
        "pytest_add_cli_args=-m\n    not e2e\n    -p\n    no:cacheprovider\n"
        "pytest_add_cli_args_test_selection=tests\n"
    )


def _mutmut(workdir: Path, *args: str, log: Path) -> int:
    with log.open("a") as out:
        return subprocess.run(
            [sys.executable, "-m", "mutmut", *args],
            cwd=workdir,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def score(target: Target, workroot: Path) -> dict[str, Any]:
    """Mutate one package in a copy; return mutmut's counts plus the survivors."""
    workdir = workroot / target.package
    shutil.copytree(REPO_ROOT / "packages" / target.package, workdir, ignore=_COPY_IGNORE)
    (workdir / "setup.cfg").write_text(_setup_cfg(target))
    log = workroot / f"{target.package}.log"
    code = _mutmut(workdir, "run", log=log)
    if code != 0:
        tail = log.read_text().splitlines()[-40:]
        raise SystemExit(
            f"mutmut run failed for {target.package} (exit {code}); the suite must pass "
            "unmutated first. Last lines:\n" + "\n".join(tail)
        )
    _mutmut(workdir, "export-cicd-stats", log=log)
    stats: dict[str, Any] = json.loads((workdir / "mutants" / "mutmut-cicd-stats.json").read_text())
    listing = subprocess.run(
        [sys.executable, "-m", "mutmut", "results"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    stats["survivors"] = sorted(
        line.split(":")[0].strip() for line in listing.splitlines() if line.endswith(": survived")
    )
    return stats


def compare(results: dict[str, dict[str, Any]], baseline: dict[str, int]) -> list[str]:
    """Failures for every target with more survivors than its baseline."""
    return [
        f"{package}: {stats['survived']} survivors, baseline {baseline.get(package)}"
        for package, stats in results.items()
        if package in baseline and int(stats["survived"]) > baseline[package]
    ]


def render(results: dict[str, dict[str, Any]], baseline: dict[str, int]) -> str:
    lines = [
        "## Mutation score",
        "",
        "| package | killed | survived | baseline | no tests | timeout | total |",
        "|---|---|---|---|---|---|---|",
    ]
    for package, stats in results.items():
        lines.append(
            f"| {package} | {stats['killed']} | {stats['survived']} | "
            f"{baseline.get(package, '-')} | {stats['no_tests']} | {stats['timeout']} | "
            f"{stats['total']} |"
        )
    for package, stats in results.items():
        survivors = stats["survivors"]
        if survivors:
            lines += ["", f"<details><summary>{package} survivors</summary>", "", "```"]
            lines += survivors
            lines += ["```", "", "</details>"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="write this run's survivor counts as the new baseline",
    )
    parser.add_argument(
        "--package", action="append", help="limit the run to these packages (repeatable)"
    )
    args = parser.parse_args(argv)
    baseline: dict[str, int] = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}
    targets = [t for t in TARGETS if not args.package or t.package in args.package]

    with tempfile.TemporaryDirectory(prefix="mbt-mutation-") as tmp:
        results = {target.package: score(target, Path(tmp)) for target in targets}

    report = render(results, baseline)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report)

    if args.update_baseline:
        updated = {**baseline, **{p: int(s["survived"]) for p, s in results.items()}}
        BASELINE.write_text(json.dumps(dict(sorted(updated.items())), indent=2) + "\n")
        print(f"wrote {BASELINE.relative_to(REPO_ROOT)}")
        return 0

    failures = compare(results, baseline)
    if failures:
        print("mutation score regressed (more survivors than the baseline):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    lower = [p for p, s in results.items() if p in baseline and int(s["survived"]) < baseline[p]]
    if lower:
        print(
            f"ratchet: {', '.join(lower)} improved; lower the baseline with --update-baseline "
            "and commit it"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
