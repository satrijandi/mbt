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

CI runs one job per module and adds the results up (``--plan``, then
``--module`` per job, then ``--combine``). Mutated whole on one runner, the
two packages took 92-120 minutes and outran the job's two-hour limit on seven
of eleven nights, and a cancelled run measures nothing. A module is the finest
split mutmut's config expresses, and it is exact: a mutant runs only the tests
that cover its own function, so the per-module counts sum to what one run over
the package reports. Locally each package still runs whole, which collects
mutmut's test stats once rather than once per module.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "scripts" / "mutation_baseline.json"

#: Exit code for a ratchet failure: a quality verdict, as everywhere in mbt
#: (1 is a hard error - here, a harness that could not measure).
REGRESSED = 2

#: How often a running mutmut reports its progress. Its own status line is
#: redrawn in place with carriage returns, which a CI log cannot show, so a
#: job that hits its time limit would otherwise leave no trace of how far it got.
HEARTBEAT_S = 300.0


@dataclass(frozen=True)
class Target:
    """One workspace package, and the modules in it worth mutating."""

    package: str
    only_mutate: tuple[str, ...]

    @property
    def root(self) -> Path:
        return REPO_ROOT / "packages" / self.package

    def modules(self) -> tuple[str, ...]:
        """The files ``only_mutate`` matches that have anything to mutate,
        as package-relative paths."""
        return tuple(
            sorted(
                {
                    path.relative_to(self.root).as_posix()
                    for pattern in self.only_mutate
                    for path in self.root.glob(pattern)
                    if path.suffix == ".py" and _has_functions(path)
                }
            )
        )


#: The code that decides whether a model ships or a batch alarms (C-1): gates,
#: the verdict, monitors and checks; the seed ladder every seeded stage reads;
#: and the metrics and paired bootstrap the gates compare.
TARGETS = (
    Target("mbt-core", ("src/mbt/quality/*", "src/mbt/execute/seeds.py")),
    Target("mbt-adapter-base", ("src/mbt_adapter_base/metrics.py",)),
)

#: mutmut's per-run counts, all of which sum across modules.
COUNTS = (
    "killed",
    "survived",
    "no_tests",
    "timeout",
    "suspicious",
    "skipped",
    "segfault",
    "check_was_interrupted_by_user",
    "total",
)

_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "mutants", ".pytest_cache", ".hypothesis", "*.pyc"
)


def _has_functions(path: Path) -> bool:
    """mutmut mutates only top-level functions and the methods of top-level
    classes, so a module of constants (``quality/check_names.py``) has no
    mutants and a job for it would measure nothing."""
    for node in ast.parse(path.read_text()).body:
        body = node.body if isinstance(node, ast.ClassDef) else [node]
        if any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) for child in body):
            return True
    return False


def plan(targets: list[Target]) -> dict[str, list[dict[str, str]]]:
    """The CI job matrix: one entry per module, so a new one gets a job."""
    return {
        "include": [
            {"package": target.package, "module": module}
            for target in targets
            for module in target.modules()
        ]
    }


def _setup_cfg(modules: tuple[str, ...]) -> str:
    only = "\n    ".join(modules)
    return (
        "[mutmut]\n"
        "source_paths=src\n"
        f"only_mutate={only}\n"
        "pytest_add_cli_args=-m\n    not e2e\n    -p\n    no:cacheprovider\n"
        "pytest_add_cli_args_test_selection=tests\n"
    )


def _progress(log: Path) -> str:
    """The newest of mutmut's redrawn status lines, without the spinner."""
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 4096))
        tail = handle.read().decode(errors="replace")
    lines = [line.strip() for line in re.split(r"[\r\n]", tail) if line.strip()]
    return lines[-1].lstrip("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ") if lines else "starting"


def _mutmut(workdir: Path, *args: str, log: Path, label: str) -> int:
    started = time.monotonic()
    with log.open("a") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mutmut", *args],
            cwd=workdir,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                return proc.wait(timeout=HEARTBEAT_S)
            except subprocess.TimeoutExpired:
                minutes = (time.monotonic() - started) / 60
                print(f"{label}: {minutes:.0f} min, {_progress(log)}", flush=True)


def score(target: Target, modules: tuple[str, ...], workroot: Path) -> dict[str, Any]:
    """Mutate ``modules`` of one package in a copy; return mutmut's counts plus
    the survivors, labelled with what was mutated."""
    workdir = workroot / target.package
    shutil.copytree(target.root, workdir, ignore=_COPY_IGNORE)
    (workdir / "setup.cfg").write_text(_setup_cfg(modules))
    log = workroot / f"{target.package}.log"
    label = f"{target.package} {modules[0] if len(modules) == 1 else f'({len(modules)} modules)'}"
    started = time.monotonic()
    code = _mutmut(workdir, "run", log=log, label=label)
    if code != 0:
        tail = log.read_text().splitlines()[-40:]
        raise SystemExit(
            f"mutmut run failed for {label} (exit {code}); the suite must pass "
            "unmutated first. Last lines:\n" + "\n".join(tail)
        )
    _mutmut(workdir, "export-cicd-stats", log=log, label=label)
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
    stats["package"] = target.package
    stats["modules"] = list(modules)
    stats["minutes"] = round((time.monotonic() - started) / 60, 1)
    print(f"{label}: {stats['minutes']} min, {stats['survived']} of {stats['total']} survived")
    return stats


def combine(units: list[dict[str, Any]], targets: list[Target]) -> dict[str, dict[str, Any]]:
    """Add up per-module results per package. A partial set is refused: a
    module whose job never reported would read as fewer survivors, and the
    ratchet would pass on a run that measured less."""
    known = {target.package for target in targets}
    strays = sorted({unit["package"] for unit in units} - known)
    if strays:
        raise SystemExit(f"results for packages this run does not mutate: {strays}")
    results: dict[str, dict[str, Any]] = {}
    for target in targets:
        mine = [unit for unit in units if unit["package"] == target.package]
        covered = sorted(module for unit in mine for module in unit["modules"])
        expected = list(target.modules())
        if covered != expected:
            raise SystemExit(
                f"{target.package}: the results cover {covered}, but the modules to "
                f"mutate are {expected}; a job is missing or reported twice"
            )
        package: dict[str, Any] = {key: sum(int(unit[key]) for unit in mine) for key in COUNTS}
        package["survivors"] = sorted(s for unit in mine for s in unit["survivors"])
        results[target.package] = package
    return results


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
    parser.add_argument(
        "--plan", action="store_true", help="print the CI job matrix (one job per module) as JSON"
    )
    parser.add_argument(
        "--module", help="mutate only this module of the one --package, writing --out (a CI job)"
    )
    parser.add_argument("--out", type=Path, help="where --module writes its result")
    parser.add_argument(
        "--combine", type=Path, help="score the per-module results in this directory"
    )
    args = parser.parse_args(argv)
    targets = [t for t in TARGETS if not args.package or t.package in args.package]

    if args.plan:
        print(json.dumps(plan(targets), separators=(",", ":")))
        return 0

    if args.module:
        if len(targets) != 1 or args.out is None:
            parser.error("--module takes exactly one --package, and --out")
        (target,) = targets
        if args.module not in target.modules():
            parser.error(f"{args.module} is not a module to mutate in {target.package}")
        with tempfile.TemporaryDirectory(prefix="mbt-mutation-") as tmp:
            unit = score(target, (args.module,), Path(tmp))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(unit, indent=2) + "\n")
        return 0

    if args.combine:
        units = [json.loads(path.read_text()) for path in sorted(args.combine.glob("*.json"))]
    else:
        with tempfile.TemporaryDirectory(prefix="mbt-mutation-") as tmp:
            units = [score(target, target.modules(), Path(tmp)) for target in targets]
    results = combine(units, targets)
    baseline: dict[str, int] = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}

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
        return REGRESSED
    lower = [p for p, s in results.items() if p in baseline and int(s["survived"]) < baseline[p]]
    if lower:
        print(
            f"ratchet: {', '.join(lower)} improved; lower the baseline with --update-baseline "
            "and commit it"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
