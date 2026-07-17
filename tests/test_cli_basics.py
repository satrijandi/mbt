"""CLI behavior tests: init scaffold, schemas, non-interactive safety,
uniform outputs (S1-06/07/08, S3-08, S5-07)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt

import mbt


@pytest.fixture()
def scaffold(tmp_path: Path) -> Path:
    """mbt init + sample data, with HOME sandboxed (profiles in <tmp>/home/.mbt)."""
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home)}
    proc = subprocess.run(
        [sys.executable, "-m", "mbt.cli.main", "init", "quickstart"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    project = tmp_path / "quickstart"
    data = subprocess.run(
        [sys.executable, str(project / "scripts" / "generate_sample_data.py"), "400"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert data.returncode == 0, data.stderr
    return project


def test_init_scaffold_is_complete_and_parses(scaffold: Path, tmp_path: Path) -> None:
    for expected in (
        "mbt_project.yml",
        "profiles.yml",
        "sources.yml",
        ".gitignore",
        ".pre-commit-config.yaml",
        "renovate.json",
        "requirements.in",
        "requirements.txt",
        "CODEOWNERS",
        "promotions.yml",
        "packages.yml",
        "datasets/churn_training_set.yml",
        "models/churn_classifier.yml",
        "scoring/churn_scoring.yml",
        "tests/test_data_quality.py",
        "macros/helpers.jinja",
        ".github/workflows/pr_check.yml",
        ".github/workflows/prod_build.yml",
        ".github/workflows/promote.yml",
        ".github/workflows/scheduled_retrain.yml",
        ".github/workflows/scheduled_score.yml",
        ".github/workflows/scheduled_monitor.yml",
        "scripts/generate_sample_data.py",
        "scripts/pr_comment.js",
        "scripts/publish_state.sh",
        "scripts/fetch_state.sh",
        "README.md",
    ):
        assert (scaffold / expected).is_file(), f"scaffold missing {expected}"
    # profiles.yml installed to ~/.mbt too (TSD §18) and gitignored locally
    assert (tmp_path / "home" / ".mbt" / "profiles.yml").is_file()
    assert "profiles.yml" in (scaffold / ".gitignore").read_text()
    # project name substituted everywhere
    assert "__PROJECT_NAME__" not in (scaffold / "mbt_project.yml").read_text()

    run_mbt(["parse"], scaffold)  # parses out of the box (S1-07)


def test_scaffold_ci_installs_are_pinned(scaffold: Path) -> None:
    """Reference workflows install the pinned toolchain, never bare package
    names, so the training environment cannot float across runs (NFR-01)."""
    workflows = sorted((scaffold / ".github" / "workflows").glob("*.yml"))
    assert workflows, "scaffold has no CI workflows"
    for workflow in workflows:
        text = workflow.read_text()
        assert "pip install mbt-" not in text, f"{workflow.name} installs unpinned"
        assert "pip install -r requirements.txt" in text, workflow.name
    for name in ("requirements.in", "requirements.txt"):
        pins = (scaffold / name).read_text()
        assert "__MBT_VERSION__" not in pins, f"{name} kept the version token"
        for package in ("mbt-core", "mbt-xgboost", "mbt-mlflow"):
            # Pinned to an immutable release tag (reproducible, non-floating) yet
            # installable from a fresh checkout without a private index.
            ref = (
                f"{package} @ git+https://github.com/satrijandi/mbt"
                f"@v{mbt.__version__}#subdirectory=packages/{package}"
            )
            assert ref in pins, f"{name}: {package} not pinned to the release tag"


def test_scaffold_operational_guardrails(scaffold: Path) -> None:
    """Scheduled retrain and prod build alert on failure; the prod manifest
    persists as a durable state baseline out of the box (FR-STATE-03)."""
    workflows_dir = scaffold / ".github" / "workflows"
    for scheduled in ("scheduled_retrain.yml", "scheduled_score.yml", "scheduled_monitor.yml"):
        text = (workflows_dir / scheduled).read_text()
        assert "if: failure()" in text and "MBT_ALERT_WEBHOOK" in text, scheduled
    prod = (workflows_dir / "prod_build.yml").read_text()
    assert "if: failure()" in prod and "MBT_ALERT_WEBHOOK" in prod
    # the baseline survives the runner on the mbt-state branch (FR-STATE-03)
    pr = (scaffold / ".github" / "workflows" / "pr_check.yml").read_text()
    assert "publish_state.sh" in prod
    assert "fetch_state.sh" in prod and "fetch_state.sh" in pr


def test_init_template_validates_against_published_schemas(scaffold: Path) -> None:
    import jsonschema
    import yaml

    run_mbt(["parse", "--write-json-schema"], scaffold)
    schema_dir = scaffold / "target" / "json-schemas"
    pairs = [
        ("sources.yml", "sources.schema.json"),
        ("datasets/churn_training_set.yml", "datasets.schema.json"),
        ("models/churn_classifier.yml", "models.schema.json"),
        ("scoring/churn_scoring.yml", "scoring.schema.json"),
        ("mbt_project.yml", "mbt_project.schema.json"),
    ]
    for spec_file, schema_file in pairs:
        payload = yaml.safe_load((scaffold / spec_file).read_text())
        schema = json.loads((schema_dir / schema_file).read_text())
        jsonschema.validate(payload, schema)  # S1-08


def test_commands_are_non_interactive_with_stdin_detached(scaffold: Path) -> None:
    # run_mbt always detaches stdin; a prompt would hang or crash (FR-CLI-01)
    run_mbt(["ls", "--output", "name"], scaffold)
    run_mbt(["show", "churn_classifier", "--output", "json"], scaffold)
    run_mbt(["clean"], scaffold)


def test_version_flag_reports_version(tmp_path: Path) -> None:
    # `mbt --version` is a standard affordance (bug reports, CI); it must work
    # without a project - the eager callback fires before any project loads.
    from mbt import __version__

    out = run_mbt(["--version"], tmp_path).stdout
    assert out.strip() == f"mbt {__version__}"


def test_ls_outputs(scaffold: Path) -> None:
    names = run_mbt(["ls", "--output", "name"], scaffold).stdout.split()
    assert "churn_classifier" in names and "churn_training_set" in names

    payload = json.loads(run_mbt(["ls", "--output", "json"], scaffold).stdout)
    by_name = {entry["name"]: entry for entry in payload}
    assert by_name["churn_classifier"]["resource_type"] == "model"
    assert "weekly" in by_name["churn_classifier"]["tags"]

    only_models = json.loads(
        run_mbt(["ls", "--select", "resource_type:model", "--output", "json"], scaffold).stdout
    )
    assert {e["resource_type"] for e in only_models} == {"model"}


def test_show_renders_compiled_config(scaffold: Path) -> None:
    payload = json.loads(run_mbt(["show", "churn_classifier", "--output", "json"], scaffold).stdout)
    assert payload["config"]["evaluation"]["gates"][0]["threshold"] == 0.25  # var resolved
    assert payload["config_hash"].startswith("sha256:")

    proc = run_mbt(["show", "churn_classifer"], scaffold, expect_exit=1)
    assert "did you mean" in proc.stderr


def test_run_operation_renders_macro(scaffold: Path) -> None:
    proc = run_mbt(["run-operation", "recent_window", "--args", "days: 14"], scaffold)
    assert proc.stdout.strip() == "-14d:now"
    missing = run_mbt(["run-operation", "nope"], scaffold, expect_exit=1)
    assert "unknown macro" in missing.stderr


def test_parse_error_exits_1_with_all_errors(scaffold: Path) -> None:
    model = scaffold / "models" / "churn_classifier.yml"
    model.write_text(model.read_text().replace("seed: 42", "sead: 42"))
    proc = run_mbt(["parse"], scaffold, expect_exit=1)
    combined = proc.stdout + proc.stderr
    assert "sead" in combined and "did you mean" in combined
    assert "seed" in combined  # missing required field also reported


@pytest.mark.e2e
def test_scaffold_state_branch_loop_end_to_end(scaffold: Path) -> None:
    """The durable-state loop the reference workflows run, executed for real:
    prod build -> publish_state.sh -> mbt-state branch on origin ->
    fetch_state.sh -> state diff flags exactly the edited model (G3,
    FR-STATE-03). Deep snapshots throughout, exactly like the workflows:
    a fresh checkout (new mtimes, same bytes) must diff empty (ADR-11).
    No GitHub required: origin is a local bare repo."""
    import shutil

    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)

    def sh(cwd: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(cwd / "scripts" / script), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    origin = scaffold.parent / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(origin)], capture_output=True, check=True
    )
    git(scaffold, "init", "--quiet")
    git(scaffold, "remote", "add", "origin", str(origin))
    git(scaffold, "config", "user.name", "test")
    git(scaffold, "config", "user.email", "test@example.com")
    git(scaffold, "add", "-A")
    git(scaffold, "commit", "--quiet", "-m", "initial")

    # bootstrap: no baseline published yet -> fetch signals exit 3
    fetch = sh(scaffold, "fetch_state.sh")
    assert fetch.returncode == 3, fetch.stdout + fetch.stderr
    # and publishing without a manifest fails loudly
    publish = sh(scaffold, "publish_state.sh")
    assert publish.returncode == 1 and "not found" in publish.stderr

    # first prod build (full: no baseline), then publish the baseline
    run_mbt(["build", "--target", "prod", "--deep-snapshot"], scaffold)
    publish = sh(scaffold, "publish_state.sh")
    assert publish.returncode == 0, publish.stdout + publish.stderr
    # plumbing-only contract: the project repo's index and tree stay clean
    assert git(scaffold, "status", "--porcelain").stdout.strip() == ""

    # the PR-check side, in a simulated fresh checkout: same bytes, new
    # mtimes. Deep snapshots keep the diff empty; the default mtime scheme
    # would flag every dataset here and retrain the world on every CI run.
    checkout = scaffold.parent / "fresh_checkout"
    # copy_function=copy: fresh mtimes, same bytes - like actions/checkout
    # (copytree's default copy2 would preserve mtimes and prove nothing)
    shutil.copytree(scaffold, checkout, copy_function=shutil.copy)
    fetch = sh(checkout, "fetch_state.sh")
    assert fetch.returncode == 0, fetch.stdout + fetch.stderr
    diff = json.loads(
        run_mbt(
            [
                "state",
                "diff",
                "--state",
                "state/prod/latest.json",
                "--deep-snapshot",
                "--output",
                "json",
            ],
            checkout,
        ).stdout
    )
    assert diff["modified"] == [] and diff["added"] == [], diff

    # edit the model spec in the checkout -> exactly that model is flagged
    model = checkout / "models" / "churn_classifier.yml"
    model.write_text(model.read_text().replace("max_depth: 4", "max_depth: 6"))
    diff = json.loads(
        run_mbt(
            [
                "state",
                "diff",
                "--state",
                "state/prod/latest.json",
                "--deep-snapshot",
                "--output",
                "json",
            ],
            checkout,
        ).stdout
    )
    modified = {entry["unique_id"] for entry in diff["modified"]}
    assert any(uid.endswith("churn_classifier") for uid in modified), diff
    assert all("churn_training_set" not in uid for uid in modified), diff

    # a second publish appends to the audit trail instead of rewriting it
    run_mbt(["compile", "--target", "prod", "--deep-snapshot"], checkout)
    publish = sh(checkout, "publish_state.sh")
    assert publish.returncode == 0, publish.stdout + publish.stderr
    history = subprocess.run(
        ["git", "-C", str(origin), "rev-list", "--count", "mbt-state"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert history.stdout.strip() == "2"
    # and the fetched baseline now matches the newly published manifest
    fetch = sh(checkout, "fetch_state.sh")
    assert fetch.returncode == 0
    fetched = (checkout / "state" / "prod" / "latest.json").read_bytes()
    assert fetched == (checkout / "target" / "manifest.json").read_bytes()


@pytest.mark.e2e
def test_project_dir_from_foreign_cwd_confines_writes_to_project(
    scaffold: Path, tmp_path: Path
) -> None:
    """`mbt --project-dir X` invoked from an unrelated cwd must confine every
    write (target/, artifact store, sqlite registry) to the project, find the
    project's registry for promote, and treat typed paths as shell-relative.
    Root cause of the FEEDBACK 2.6 litter: config-relative paths resolved
    against the process cwd; the coordinator now chdirs to the project like
    job subprocesses always did."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def mbt_from_elsewhere(*args: str) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [sys.executable, "-m", "mbt.cli.main", *args, "--project-dir", str(scaffold)],
            cwd=elsewhere,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        assert proc.returncode == 0, f"mbt {' '.join(args)}:\n{proc.stdout}\n{proc.stderr}"
        return proc

    mbt_from_elsewhere("build", "--target", "prod")
    assert (scaffold / "target" / "artifacts").is_dir()  # store under the project
    assert (scaffold / "mlflow.db").is_file()  # relative sqlite uri under the project
    assert list(elsewhere.iterdir()) == [], "build leaked writes into the invocation cwd"

    # promote must find the model in the PROJECT's registry; a cwd-relative
    # sqlite uri used to open an empty db in the invocation cwd instead
    mbt_from_elsewhere("promote", "--model", "churn_classifier", "--to", "production")
    assert list(elsewhere.iterdir()) == [], "promote leaked writes into the invocation cwd"

    # paths TYPED on the command line stay shell-relative to the invocation cwd
    baseline = elsewhere / "baseline.json"
    baseline.write_bytes((scaffold / "target" / "manifest.json").read_bytes())
    diff_out = mbt_from_elsewhere("state", "diff", "--state", "./baseline.json", "--output", "json")
    assert json.loads(diff_out.stdout)["modified"] == []
    assert list(elsewhere.iterdir()) == [baseline]


def test_unknown_option_prints_clean_usage_error(scaffold: Path) -> None:
    """A mistyped flag must exit 1 with click's usage error, never a raw
    traceback (typer vendors click, so main() must catch the vendored
    exception types too)."""
    proc = run_mbt(["state", "diff", "--no-such-flag"], scaffold, expect_exit=1)
    assert "No such option" in proc.stderr
    assert "Traceback" not in proc.stderr and "Traceback" not in proc.stdout


def test_state_diff_deep_snapshot_ignores_mtime_churn(scaffold: Path) -> None:
    """The runbook remedy for mtime false-positives: deep-snapshot manifests
    diffed with --deep-snapshot survive a touch; the default mtime scheme
    flags it (ADR-11)."""
    manifest = scaffold / "target" / "manifest.json"
    run_mbt(["compile", "--anchor", DEMO_ANCHOR, "--deep-snapshot"], scaffold)
    baseline_deep = scaffold / "target" / "manifest_deep.json"
    baseline_deep.write_bytes(manifest.read_bytes())
    run_mbt(["compile", "--anchor", DEMO_ANCHOR], scaffold)
    baseline_mtime = scaffold / "target" / "manifest_mtime.json"
    baseline_mtime.write_bytes(manifest.read_bytes())

    # the training source specifically: churn_outcomes sorts first but is a
    # ground-truth label table, deliberately outside node identity (ADR-20)
    parquet = next((scaffold / "data" / "subscribers").rglob("*.parquet"))
    os.utime(parquet)  # mtime changes, bytes do not

    deep = json.loads(
        run_mbt(
            [
                "state",
                "diff",
                "--state",
                str(baseline_deep),
                "--anchor",
                DEMO_ANCHOR,
                "--deep-snapshot",
                "--output",
                "json",
            ],
            scaffold,
        ).stdout
    )
    assert deep["modified"] == [], deep

    mtime = json.loads(
        run_mbt(
            [
                "state",
                "diff",
                "--state",
                str(baseline_mtime),
                "--anchor",
                DEMO_ANCHOR,
                "--output",
                "json",
            ],
            scaffold,
        ).stdout
    )
    assert any("snapshot" in e["components"] for e in mtime["modified"]), mtime


def test_state_selector_without_state_flag_fails_loudly(scaffold: Path) -> None:
    (scaffold / "data" / "subscribers").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(scaffold / "scripts" / "generate_sample_data.py"), "300"],
        cwd=scaffold,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    failed = run_mbt(
        ["build", "--select", "state:modified", "--anchor", DEMO_ANCHOR],
        scaffold,
        expect_exit=1,
    )
    assert "--state" in failed.stderr
