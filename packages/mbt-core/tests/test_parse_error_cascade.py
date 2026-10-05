"""One broken resource is one error, not one per resource that refs it
(FEEDBACK v6 B-3).

A model that failed validation used to vanish from the link table, so every
scoring pipeline that ref'd it added "references unknown model" - a false
error about a resource that exists - and two errors for one mistake teach
people to distrust the error list.
"""

from pathlib import Path

from mbt.adapters.registry import AdapterRegistry
from mbt.parsing import parse_project

DATASET = "datasets/churn_training.yml"


def _messages(project: Path, registry: AdapterRegistry) -> list[str]:
    parsed = parse_project(project, registry=registry, raise_on_error=False)
    return [issue.format() for issue in parsed.report.errors]


def test_an_invalid_dataset_is_not_also_an_unknown_ref(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    path = demo_project / DATASET
    path.write_text(path.read_text().replace("strategy: temporal", "strategy: temporl"))
    (message,) = _messages(demo_project, fake_registry)
    assert "/split" in message and "unknown dataset" not in message


def test_a_file_refused_for_a_duplicate_key_is_not_also_an_unknown_ref(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    path = demo_project / DATASET
    path.write_text(path.read_text().replace("tags: [churn]", "tags: [churn]\n    tags: [x]"))
    (message,) = _messages(demo_project, fake_registry)
    assert "found duplicate key 'tags'" in message


def test_a_file_that_is_not_yaml_is_named_in_the_dangling_ref_hint(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    (demo_project / DATASET).write_text("datasets:\n  - name: [\n")
    messages = _messages(demo_project, fake_registry)
    assert len(messages) == 2  # the file, and a ref nothing can vouch for
    dangling = next(m for m in messages if "unknown dataset" in m)
    assert f"it may be declared in {DATASET}, which could not be read" in dangling


def test_a_real_typo_in_a_ref_is_still_reported(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    path = demo_project / "models/churn_model.yml"
    path.write_text(path.read_text().replace("ref('churn_training')", "ref('churn_trainng')"))
    (message,) = _messages(demo_project, fake_registry)
    assert "unknown dataset ref('churn_trainng')" in message
    assert "did you mean 'churn_training'?" in message


def test_scoring_and_exposures_do_not_echo_a_broken_model(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    from core_helpers import write
    from test_scoring_parsing import SCORING_YML

    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: scoring_batch
                path: data/scoring_batch/*.parquet
              - name: churn_outcomes
                path: data/churn_outcomes/*.parquet
        """,
    )
    write(demo_project / "scoring/churn_scoring.yml", SCORING_YML)
    write(
        demo_project / "exposures.yml",
        """
        exposures:
          - name: dash
            type: dashboard
            owner: bi@example.com
            depends_on: ["ref('churn_model')"]
        """,
    )
    path = demo_project / "models/churn_model.yml"
    path.write_text(path.read_text().replace("task: binary_classification", "task: nope"))
    (message,) = _messages(demo_project, fake_registry)
    assert "/task" in message


def test_a_file_whose_yaml_is_not_a_mapping_names_nothing(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    (demo_project / DATASET).write_text("- just\n- a list\n")
    dangling = next(m for m in _messages(demo_project, fake_registry) if "unknown dataset" in m)
    assert f"it may be declared in {DATASET}" in dangling
