"""CLI behavior tests: init scaffold, schemas, non-interactive safety,
uniform outputs (S1-06/07/08, S3-08, S5-07)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from e2e_utils import DEMO_ANCHOR, run_mbt


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
        "CODEOWNERS",
        "promotions.yml",
        "packages.yml",
        "datasets/churn_training_set.yml",
        "models/churn_classifier.yml",
        "tests/test_data_quality.py",
        "macros/helpers.jinja",
        ".github/workflows/pr_check.yml",
        ".github/workflows/prod_build.yml",
        ".github/workflows/promote.yml",
        ".github/workflows/scheduled_retrain.yml",
        "scripts/generate_sample_data.py",
        "scripts/pr_comment.js",
        "README.md",
    ):
        assert (scaffold / expected).is_file(), f"scaffold missing {expected}"
    # profiles.yml installed to ~/.mbt too (TSD §18) and gitignored locally
    assert (tmp_path / "home" / ".mbt" / "profiles.yml").is_file()
    assert "profiles.yml" in (scaffold / ".gitignore").read_text()
    # project name substituted everywhere
    assert "__PROJECT_NAME__" not in (scaffold / "mbt_project.yml").read_text()

    run_mbt(["parse"], scaffold)  # parses out of the box (S1-07)


def test_init_template_validates_against_published_schemas(scaffold: Path) -> None:
    import jsonschema
    import yaml

    run_mbt(["parse", "--write-json-schema"], scaffold)
    schema_dir = scaffold / "target" / "json-schemas"
    pairs = [
        ("sources.yml", "sources.schema.json"),
        ("datasets/churn_training_set.yml", "datasets.schema.json"),
        ("models/churn_classifier.yml", "models.schema.json"),
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
    assert payload["config"]["evaluation"]["gates"][0]["threshold"] == 0.05  # var resolved
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
