"""The nightly mutation tier (FEEDBACK v6 C-1): the harness's ratchet, and the
workflow that runs it.

The harness itself takes tens of minutes, so what is pinned here is the part
that decides pass/fail and the wiring that would let a red run go unseen.
"""

import importlib.util
import json
from types import ModuleType

import pytest
import yaml
from e2e_utils import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "mutation.yml"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    path = REPO_ROOT / "scripts" / "mutation_score.py"
    spec = importlib.util.spec_from_file_location("mutation_score", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stats(survived: int) -> dict[str, object]:
    return {
        "killed": 100,
        "survived": survived,
        "no_tests": 0,
        "timeout": 0,
        "total": 100 + survived,
        "survivors": [f"m.x__mutmut_{i}" for i in range(survived)],
    }


def test_the_ratchet_fails_only_on_more_survivors(harness: ModuleType) -> None:
    baseline = {"mbt-core": 10}
    assert harness.compare({"mbt-core": _stats(10)}, baseline) == []
    assert harness.compare({"mbt-core": _stats(9)}, baseline) == []
    (failure,) = harness.compare({"mbt-core": _stats(11)}, baseline)
    assert failure == "mbt-core: 11 survivors, baseline 10"


def test_the_summary_lists_counts_and_every_survivor(harness: ModuleType) -> None:
    report = harness.render({"mbt-core": _stats(2)}, {"mbt-core": 3})
    assert "| mbt-core | 100 | 2 | 3 | 0 | 0 | 102 |" in report
    assert "m.x__mutmut_0" in report and "m.x__mutmut_1" in report


def test_the_targets_are_the_code_that_decides_whether_a_model_ships(
    harness: ModuleType,
) -> None:
    globs = {t.package: t.only_mutate for t in harness.TARGETS}
    assert "src/mbt/quality/*" in globs["mbt-core"]
    assert "src/mbt/execute/seeds.py" in globs["mbt-core"]
    assert globs["mbt-adapter-base"] == ("src/mbt_adapter_base/metrics.py",)
    for target in harness.TARGETS:
        package = REPO_ROOT / "packages" / target.package
        for pattern in target.only_mutate:
            assert list(package.glob(pattern)), f"{target.package}: {pattern} matches nothing"
        config = harness._setup_cfg(target)
        assert "source_paths=src" in config and "not e2e" in config


def test_every_target_has_a_committed_baseline(harness: ModuleType) -> None:
    baseline = json.loads((REPO_ROOT / "scripts" / "mutation_baseline.json").read_text())
    assert set(baseline) == {t.package for t in harness.TARGETS}
    assert all(isinstance(count, int) and count >= 0 for count in baseline.values())


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_it_runs_nightly_and_on_demand(workflow: dict) -> None:
    triggers = workflow[True]  # PyYAML reads the bare `on:` key as True
    assert triggers["schedule"] and "workflow_dispatch" in triggers


def test_it_runs_the_harness_from_the_locked_mutation_group(workflow: dict) -> None:
    steps = workflow["jobs"]["mutation"]["steps"]
    runs = [str(step.get("run", "")) for step in steps]
    assert "uv sync --group mutation" in runs
    assert "uv run --group mutation python scripts/mutation_score.py" in runs
    assert not any("--update-baseline" in run for run in runs), "CI must never move the baseline"


def test_a_red_run_files_an_issue(workflow: dict) -> None:
    assert workflow["permissions"]["issues"] == "write"
    notice = workflow["jobs"]["mutation"]["steps"][-1]
    assert notice["if"] == "always()"
    assert "gh issue create" in notice["run"] and "gh issue close" in notice["run"]


def test_the_mutation_group_stays_out_of_the_default_install() -> None:
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    groups = pyproject["dependency-groups"]
    assert any(req.startswith("mutmut") for req in groups["mutation"])
    assert not any(str(req).startswith("mutmut") for req in groups["dev"])
