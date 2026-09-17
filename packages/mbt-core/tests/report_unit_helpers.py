"""Shared builders for the training-report tests (ADR-30; unique module name)."""

from pathlib import Path

from core_helpers import TEST_ANCHOR

from mbt.adapters.registry import AdapterRegistry
from mbt.artifacts.manifest import Manifest
from mbt.compile.compiler import CompileOptions, compile_project
from mbt.config.profiles import load_profiles
from mbt.parsing import parse_project

DATASET_FILE = "datasets/churn_training.yml"
MODEL_FILE = "models/churn_model.yml"
SPLIT_BLOCK = 'train: "-180d:-28d"\n      test: "-28d:now"'


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not found in {path}"
    path.write_text(text.replace(old, new))


def with_out_of_time(project: Path, *, test: str = "-60d:-30d", after: str = "-30d:now") -> None:
    """Give the demo dataset an after-test window."""
    edit(
        project / DATASET_FILE,
        SPLIT_BLOCK,
        f'train: "-180d:-60d"\n      test: "{test}"\n      out_of_time: "{after}"',
    )


def model_evaluation(project: Path, extra: str) -> None:
    """Append lines under the demo model's ``evaluation:`` block."""
    edit(
        project / MODEL_FILE,
        "      metrics: [pr_auc, roc_auc]\n",
        "      metrics: [pr_auc, roc_auc]\n" + extra,
    )


def add_gate(project: Path, gate: str) -> None:
    """Append one gate (a flow mapping) to the demo model's gates."""
    edit(
        project / MODEL_FILE,
        "          threshold: \"{{ var('default_threshold') }}\"\n",
        "          threshold: \"{{ var('default_threshold') }}\"\n" + f"        - {gate}\n",
    )


REPORT_CONFIG = (
    "      stability:\n"
    "        min_rows: 5\n"
    "        prediction_shift: {method: psi, threshold: 50}\n"
    "        feature_shift: {method: psi, threshold: 50}\n"
    "      report:\n"
    "        predictions: {enabled: true, max_rows: 20}\n"
    "        binning: all\n"
    "        min_rows: 5\n"
)

AFTER_TEST_GATE = "{metric: roc_auc, threshold: 0.0, source: out_of_time, min_rows: 5}"


def reporting_project(project: Path) -> Path:
    """The demo project with an after-test window, a full report, and gates."""
    with_out_of_time(project)
    model_evaluation(project, REPORT_CONFIG)
    add_gate(project, AFTER_TEST_GATE)
    return project


def compile_demo(project: Path, registry: AdapterRegistry, anchor=TEST_ANCHOR) -> Manifest:
    parsed = parse_project(project, registry=registry)
    profiles = load_profiles("demo", project, project_vars=parsed.project.vars)
    return compile_project(
        parsed, profiles, registry=registry, options=CompileOptions(anchor=anchor)
    )
