"""The nightly mutation tier (FEEDBACK v6 C-1): the harness's ratchet, and the
workflow that runs it.

The harness itself takes tens of minutes, so what is pinned here is the part
that decides pass/fail and the wiring that would let a red run go unseen.
"""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
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


def _unit(package: str, module: str, survived: int) -> dict[str, object]:
    """One mutate job's result, as `--module ... --out` writes it."""
    counts = dict.fromkeys(
        ("suspicious", "skipped", "segfault", "check_was_interrupted_by_user"), 0
    )
    return {
        **_stats(survived),
        **counts,
        "survivors": [f"{module}.x__mutmut_{i}" for i in range(survived)],
        "package": package,
        "modules": [module],
        "minutes": 1.0,
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
        config = harness._setup_cfg(target.modules())
        assert "source_paths=src" in config and "not e2e" in config


def test_every_target_has_a_committed_baseline(harness: ModuleType) -> None:
    baseline = json.loads((REPO_ROOT / "scripts" / "mutation_baseline.json").read_text())
    assert set(baseline) == {t.package for t in harness.TARGETS}
    assert all(isinstance(count, int) and count >= 0 for count in baseline.values())


# -- one job per module -----------------------------------------------------------


def test_the_plan_gives_every_module_with_code_exactly_one_job(harness: ModuleType) -> None:
    """The split CI runs. A module left out is a module nobody mutates, and
    one planned twice is counted twice."""
    jobs = harness.plan(list(harness.TARGETS))["include"]
    planned = [(job["package"], job["module"]) for job in jobs]
    assert len(planned) == len(set(planned)), planned
    for target in harness.TARGETS:
        expected = {
            path.relative_to(target.root).as_posix()
            for pattern in target.only_mutate
            for path in target.root.glob(pattern)
            if path.suffix == ".py" and harness._has_functions(path)
        }
        assert {m for p, m in planned if p == target.package} == expected
    assert ("mbt-core", "src/mbt/quality/checks.py") in planned
    assert ("mbt-core", "src/mbt/execute/seeds.py") in planned
    assert ("mbt-adapter-base", "src/mbt_adapter_base/metrics.py") in planned
    # Constants only: mutmut makes no mutant there, so no job either.
    assert ("mbt-core", "src/mbt/quality/check_names.py") not in planned
    assert ("mbt-core", "src/mbt/quality/__init__.py") not in planned


def test_a_module_has_code_to_mutate_only_in_a_function_or_method(
    harness: ModuleType, tmp_path: Path
) -> None:
    """mutmut mutates top-level functions and the methods of top-level
    classes, nothing else."""
    cases = {
        "NAMES = frozenset({'a', 'b'})\n": False,
        "class Spec:\n    name: str = 'x'\n": False,
        "def gate(x):\n    return x > 1\n": True,
        "async def fetch():\n    return 1\n": True,
        "class Gate:\n    def passed(self):\n        return True\n": True,
    }
    for index, (source, mutable) in enumerate(cases.items()):
        path = tmp_path / f"m{index}.py"
        path.write_text(source)
        assert harness._has_functions(path) is mutable, source


def test_combine_adds_the_modules_up_per_package(harness: ModuleType) -> None:
    core = harness.Target("mbt-core", ("src/mbt/execute/seeds.py", "src/mbt/quality/gates.py"))
    results = harness.combine(
        [
            _unit("mbt-core", "src/mbt/quality/gates.py", 3),
            _unit("mbt-core", "src/mbt/execute/seeds.py", 1),
        ],
        [core],
    )
    assert results["mbt-core"]["survived"] == 4
    assert results["mbt-core"]["killed"] == 200
    assert results["mbt-core"]["total"] == 204
    assert len(results["mbt-core"]["survivors"]) == 4


def test_combine_refuses_a_partial_or_doubled_set(harness: ModuleType) -> None:
    """A missing module reads as fewer survivors: the ratchet would pass on a
    run that measured less. That is the failure this refuses."""
    core = harness.Target("mbt-core", ("src/mbt/execute/seeds.py", "src/mbt/quality/gates.py"))
    gates = _unit("mbt-core", "src/mbt/quality/gates.py", 3)
    seeds = _unit("mbt-core", "src/mbt/execute/seeds.py", 1)
    with pytest.raises(SystemExit, match="a job is missing or reported twice"):
        harness.combine([gates], [core])
    with pytest.raises(SystemExit, match="a job is missing or reported twice"):
        harness.combine([gates, seeds, seeds], [core])
    with pytest.raises(SystemExit, match="does not mutate"):
        harness.combine([gates, seeds, _unit("mbt-other", "src/x.py", 0)], [core])


def _write_units(directory: Path, units: list[dict[str, object]]) -> None:
    directory.mkdir()
    for index, unit in enumerate(units):
        (directory / f"{index}.json").write_text(json.dumps(unit))


def test_combine_from_ci_results_exits_2_on_a_regression(
    harness: ModuleType, tmp_path: Path, monkeypatch, capsys
) -> None:
    """The score job's exit code is how the notice tells a regression (2,
    mbt's quality verdict) from a harness that could not score (1)."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    core = harness.Target("mbt-core", ("src/mbt/execute/seeds.py",))
    monkeypatch.setattr(harness, "TARGETS", (core,))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"mbt-core": 2}))
    monkeypatch.setattr(harness, "BASELINE", baseline)

    _write_units(tmp_path / "at", [_unit("mbt-core", "src/mbt/execute/seeds.py", 2)])
    assert harness.main(["--combine", str(tmp_path / "at")]) == 0

    _write_units(tmp_path / "over", [_unit("mbt-core", "src/mbt/execute/seeds.py", 3)])
    assert harness.main(["--combine", str(tmp_path / "over")]) == harness.REGRESSED == 2
    assert "mbt-core: 3 survivors, baseline 2" in capsys.readouterr().err

    _write_units(tmp_path / "none", [])
    with pytest.raises(SystemExit, match="a job is missing"):
        harness.main(["--combine", str(tmp_path / "none")])


def test_the_plan_prints_one_line_of_json(harness: ModuleType, capsys) -> None:
    """It goes into $GITHUB_OUTPUT as `matrix=...`, which ends at a newline."""
    assert harness.main(["--plan"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert json.loads(out) == harness.plan(list(harness.TARGETS))


def test_a_module_job_names_one_planned_module(harness: ModuleType, tmp_path: Path) -> None:
    out = str(tmp_path / "r.json")
    with pytest.raises(SystemExit):
        harness.main(["--module", "src/mbt/quality/gates.py", "--out", out])
    with pytest.raises(SystemExit):
        harness.main(["--package", "mbt-core", "--module", "src/mbt/quality/gates.py"])
    with pytest.raises(SystemExit):
        harness.main(
            ["--package", "mbt-core", "--module", "src/mbt/quality/check_names.py", "--out", out]
        )


def test_progress_is_mutmuts_newest_status_line(harness: ModuleType, tmp_path: Path) -> None:
    """mutmut redraws its status with carriage returns, which a CI log shows
    as nothing until the process exits - and a timed-out job never exits."""
    log = tmp_path / "mutmut.log"
    log.write_text(
        "    done in 812ms\n"
        "\r⠋ 10/2688  🎉 9 🫥 0  ⏰ 0  🤔 0  🙁 1  🔇 0  🧙 0"
        "\r⠙ 1522/2688  🎉 1286 🫥 0  ⏰ 1  🤔 0  🙁 235  🔇 0  🧙 0   "
    )
    assert harness._progress(log).startswith("1522/2688  🎉 1286")
    log.write_text("")
    assert harness._progress(log) == "starting"


def test_a_long_mutmut_run_reports_progress(
    harness: ModuleType, tmp_path: Path, monkeypatch, capsys
) -> None:
    log = tmp_path / "mutmut.log"
    log.write_text("\r⠋ 7/9  🎉 6 🙁 1")

    class _Slow:
        """A mutmut that is still running at the first heartbeat."""

        waits = 0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def wait(self, timeout: float) -> int:
            type(self).waits += 1
            if type(self).waits == 1:
                raise subprocess.TimeoutExpired("mutmut", timeout)
            return 0

    monkeypatch.setattr(harness.subprocess, "Popen", _Slow)
    assert harness._mutmut(tmp_path, "run", log=log, label="mbt-core gates.py") == 0
    assert "mbt-core gates.py: 0 min, 7/9  🎉 6 🙁 1" in capsys.readouterr().out


# -- the workflow -------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_it_runs_nightly_and_on_demand(workflow: dict) -> None:
    triggers = workflow[True]  # PyYAML reads the bare `on:` key as True
    assert triggers["schedule"] and "workflow_dispatch" in triggers


def _runs(job: dict) -> list[str]:
    return [str(step.get("run", "")) for step in job["steps"]]


def test_it_mutates_one_module_per_job_from_the_locked_mutation_group(workflow: dict) -> None:
    jobs = workflow["jobs"]
    assert any("scripts/mutation_score.py --plan" in run for run in _runs(jobs["plan"]))
    assert jobs["plan"]["outputs"]["matrix"] == "${{ steps.plan.outputs.matrix }}"

    mutate = jobs["mutate"]
    assert mutate["needs"] == "plan"
    assert mutate["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.matrix) }}"
    # One slow module must not cancel the others' measurements.
    assert mutate["strategy"]["fail-fast"] is False
    assert isinstance(mutate["timeout-minutes"], int)
    runs = _runs(mutate)
    assert "uv sync --group mutation" in runs
    (harness_run,) = [run for run in runs if "scripts/mutation_score.py" in run]
    assert harness_run.startswith("uv run --group mutation python scripts/mutation_score.py")
    assert '--module "$MODULE"' in harness_run and '--out "$OUT"' in harness_run
    for name in ("plan", "mutate", "score", "notify"):
        assert not any("--update-baseline" in run for run in _runs(jobs[name])), (
            "CI must never move the baseline"
        )


def test_only_a_complete_set_is_scored(workflow: dict) -> None:
    score = workflow["jobs"]["score"]
    assert score["needs"] == "mutate"
    assert "if" not in score, "scoring after a failed or timed-out module reads as fewer survivors"
    assert any("--combine mutation-results" in run for run in _runs(score))
    download = next(s for s in score["steps"] if "download-artifact" in str(s.get("uses")))
    assert download["with"]["pattern"] == "mutation-*"
    assert download["with"]["merge-multiple"] is True


def test_a_red_run_files_an_issue(workflow: dict) -> None:
    notify = workflow["jobs"]["notify"]
    assert set(notify["needs"]) == {"plan", "mutate", "score"}
    assert notify["if"] == "always()"
    assert notify["permissions"]["issues"] == "write"
    script = _notice_script(workflow)
    assert "gh issue create" in script and "gh issue close" in script


def _notice_script(workflow: dict) -> str:
    (script,) = [run for run in _runs(workflow["jobs"]["notify"]) if run]
    return script


def _run_notice(
    script: str,
    tmp_path: Path,
    *,
    plan: str = "success",
    mutate: str = "success",
    score: str = "success",
    score_exit: str = "0",
    existing: str = "",
) -> list[str]:
    """Run the real notice script with `gh` stubbed: it logs each call, and
    answers `gh issue list` with `existing`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "gh-calls.txt"
    (bin_dir / "gh").write_text(
        "#!/usr/bin/env bash\n"
        # One line per call: an issue body spans several.
        f'printf "%s\\n" "$*" | tr "\\n" " " >> {calls}; echo >> {calls}\n'
        'if [ "$1" = "issue" ] && [ "$2" = "list" ]; then\n'
        f'  printf "%s" "{existing}"\n'
        "fi\n"
        "exit 0\n"
    )
    (bin_dir / "gh").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "stub",
        "GH_REPO": "satrijandi/mbt",
        "TITLE": "Mutation tier is failing",
        "PLAN": plan,
        "MUTATE": mutate,
        "SCORE": score,
        "SCORE_EXIT": score_exit,
        "RUN_URL": "https://example.invalid/run/1",
    }
    done = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0 and not done.stderr, done.stderr
    return calls.read_text().splitlines() if calls.exists() else []


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_a_measured_green_run_files_nothing_and_closes_the_issue(
    workflow: dict, tmp_path: Path
) -> None:
    (tmp_path / "a").mkdir()
    calls = _run_notice(_notice_script(workflow), tmp_path / "a")
    assert not [c for c in calls if c.startswith(("issue create", "issue comment"))], calls
    (tmp_path / "b").mkdir()
    calls = _run_notice(_notice_script(workflow), tmp_path / "b", existing="30")
    assert len([c for c in calls if c.startswith("issue close 30")]) == 1, calls


@needs_bash
def test_a_regression_is_reported_as_one(workflow: dict, tmp_path: Path) -> None:
    calls = _run_notice(_notice_script(workflow), tmp_path, score="failure", score_exit="2")
    (created,) = [c for c in calls if c.startswith("issue create")]
    assert "The mutation score regressed" in created


@needs_bash
def test_a_timed_out_module_is_not_reported_as_a_regression(workflow: dict, tmp_path: Path) -> None:
    """What the old notice did on seven nights: a job hit its time limit, the
    run measured nothing, and the issue said the score had regressed. A
    timed-out job's result is `cancelled`, and the score job is skipped."""
    calls = _run_notice(
        _notice_script(workflow), tmp_path, mutate="cancelled", score="skipped", score_exit=""
    )
    (created,) = [c for c in calls if c.startswith("issue create")]
    assert "did not finish" in created and "measured nothing" in created
    assert "regressed" not in created


@needs_bash
def test_a_harness_failure_names_each_job(workflow: dict, tmp_path: Path) -> None:
    for name, kwargs in {
        "mutate": {"mutate": "failure", "score": "skipped", "score_exit": ""},
        "score": {"score": "failure", "score_exit": "1"},
    }.items():
        (tmp_path / name).mkdir()
        calls = _run_notice(_notice_script(workflow), tmp_path / name, **kwargs)
        (created,) = [c for c in calls if c.startswith("issue create")]
        assert "failed before it could compare" in created, name
        assert "regressed" not in created and "did not finish" not in created, name


@needs_bash
def test_a_second_red_night_comments_rather_than_duplicating(
    workflow: dict, tmp_path: Path
) -> None:
    calls = _run_notice(
        _notice_script(workflow), tmp_path, mutate="cancelled", score="skipped", existing="31"
    )
    assert not [c for c in calls if c.startswith("issue create")], calls
    assert len([c for c in calls if c.startswith("issue comment 31")]) == 1, calls


def test_the_mutation_group_stays_out_of_the_default_install() -> None:
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    groups = pyproject["dependency-groups"]
    assert any(req.startswith("mutmut") for req in groups["mutation"])
    assert not any(str(req).startswith("mutmut") for req in groups["dev"])
