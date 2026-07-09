"""Unit tests for mbt.parsing.project_parser: accessors, error branches, linking."""

from pathlib import Path

from core_helpers import write
from parse_unit_helpers import error_messages, register_unit_plugins, warning_messages

from mbt.adapters.registry import AdapterRegistry
from mbt.parsing import parse_project


def parse(project_dir: Path, registry: AdapterRegistry):
    return parse_project(project_dir, registry=registry, raise_on_error=False)


# -- ParsedProject accessors -----------------------------------------------------


def test_accessors_names_and_tags(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    parsed = parse_project(demo_project, registry=fake_registry)
    model = parsed.models["model.demo.churn_model"]
    assert model.tags == ["churn", "weekly"]

    assert parsed.resource("model.demo.churn_model") is model  # by unique_id
    assert parsed.resource("churn_model") is model  # by name
    assert parsed.resource("churn_training").name == "churn_training"
    by_uid = parsed.resource("source.demo.lakehouse.subscribers")
    assert by_uid is parsed.resource("subscribers")  # source table by name
    assert by_uid.table.name == "subscribers"
    assert parsed.resource("no_such_thing") is None

    names = parsed.all_names()
    assert {"churn_model", "churn_training", "subscribers"} <= set(names)


# -- defaults merging and renderer setup -------------------------------------------


def test_model_defaults_dicts_merge_one_level(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "mbt_project.yml",
        """
        name: demo
        version: "1.0"
        vars:
          default_threshold: 0.4
        model_defaults:
          hyperparameters:
            learning_rate: 0.2
        """,
    )
    parsed = parse_project(demo_project, registry=fake_registry)
    merged = parsed.models["model.demo.churn_model"].raw["hyperparameters"]
    assert merged["learning_rate"] == 0.2  # from project defaults
    assert merged["max_depth"] == 4  # from the spec


def test_invalid_macro_file_is_reported_and_parse_continues(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(demo_project / "macros/broken.jinja", "{% macro oops(\n")
    parsed = parse(demo_project, fake_registry)
    assert any("invalid macro file" in m for m in error_messages(parsed))
    assert parsed.renderer.macro_names == []  # fell back to a macro-less renderer


# -- discovery -------------------------------------------------------------------


def test_broken_yaml_file_and_non_mapping_entry(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    (demo_project / "datasets/broken.yml").write_text("datasets: [unclosed\n  x: {")
    write(demo_project / "datasets/stringy.yml", "datasets: [just_a_string]\n")
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("invalid YAML" in m for m in messages)
    assert any("entry 0 under 'datasets' must be a mapping" in m for m in messages)


# -- sources and metrics -----------------------------------------------------------


def test_invalid_and_duplicate_sources(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "datasets/extra_sources.yml",
        """
        sources:
          - name: badgroup
            tables:
              - name: no_location   # neither path nor identifier
          - name: lakehouse
            tables:
              - name: subscribers   # duplicates sources.yml
                path: data/subscribers/*.parquet
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("either 'path' or 'identifier'" in m for m in messages)
    assert any("duplicate source table 'lakehouse.subscribers'" in m for m in messages)


def test_invalid_and_duplicate_metrics(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "metrics.yml",
        """
        metrics:
          - name: bad_metric
            kind: bogus_kind
          - name: dup_metric
          - name: dup_metric
        """,
    )
    parsed = parse(demo_project, fake_registry)
    assert any("duplicate metric 'dup_metric'" in m for m in error_messages(parsed))
    assert "bad_metric" not in parsed.metrics
    assert "dup_metric" in parsed.metrics


# -- datasets ---------------------------------------------------------------------


def test_dataset_error_branches(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "datasets/broken_sets.yml",
        """
        datasets:
          - name: ds_var_missing
            description: "{{ var('missing_var_zzz') }}"
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
          - name: ds_invalid
            source: source('lakehouse', 'subscribers')
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
          - name: churn_training
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
          - name: ds_badwin
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "bogus", test: "-7d:now"}
          - name: ds_badcheck
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
            checks: [not_nul]
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("var 'missing_var_zzz' has no value at parse time" in m for m in messages)
    assert any("required field 'label' is missing" in m for m in messages)
    assert any("duplicate dataset 'churn_training'" in m for m in messages)
    assert any("invalid window expression 'bogus'" in m for m in messages)
    unknown_check = [i for i in parsed.report.errors if "unknown dataset check" in i.message]
    assert unknown_check and unknown_check[0].hint == "did you mean 'not_null'?"


def test_unknown_data_test_binding(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "datasets/tested.yml",
        """
        datasets:
          - name: tested_ds
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
            tests: [ghost_test]
        """,
    )
    parsed = parse(demo_project, fake_registry)
    issues = [i for i in parsed.report.errors if "unknown data test 'ghost_test'" in i.message]
    assert issues and "(none)" in (issues[0].hint or "")


# -- models -----------------------------------------------------------------------


def test_model_capture_validation_and_hooks_branches(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "hooks/custom.py",
        "def transform_features(table, ctx):\n    return table\n",
    )
    write(
        demo_project / "models/broken_models.yml",
        """
        models:
          - name: m_var_missing
            description: "{{ var('missing_var_zzz') }}"
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_invalid
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            seed: 7
          - name: m_hooks_missing
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            hooks: hooks/nope.py
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_hooks_ok
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            hooks: hooks/custom.py
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("var 'missing_var_zzz' has no value at parse time" in m for m in messages)
    assert any("required field 'evaluation' is missing" in m for m in messages)
    assert any("hooks file 'hooks/nope.py' does not exist" in m for m in messages)
    assert parsed.models["model.demo.m_hooks_ok"].hooks_path == "hooks/custom.py"


def test_adapter_capability_error_branches(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    register_unit_plugins(fake_registry)
    write(
        demo_project / "models/capability.yml",
        """
        models:
          - name: m_notrain
            task: binary_classification
            adapter: notrain
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_wrong_task
            task: regression
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_fussy
            task: binary_classification
            adapter: fussy
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_no_schema
            task: regression
            adapter: reggy
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("adapter 'notrain' provides no training adapter" in m for m in messages)
    wrong_task = [
        i for i in parsed.report.errors if "does not support task 'regression'" in i.message
    ]
    assert wrong_task and "binary_classification" in (wrong_task[0].hint or "")
    assert any("fussy adapter says no" in m for m in messages)
    assert any("fussy adapter is uneasy" in m for m in warning_messages(parsed))
    assert any("task 'regression' has no registered task schema" in m for m in messages)


def test_tuning_engine_errors(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    register_unit_plugins(fake_registry)
    write(
        demo_project / "models/tuned.yml",
        """
        models:
          - name: m_missing_engine
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            tuning:
              engine: no_such_engine_zzz
              n_trials: 5
              search_space:
                max_depth: {type: int, low: 2, high: 6}
              objective: {metric: pr_auc, direction: maximize}
            seed: 7
          - name: m_engine_without_tuning
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            tuning:
              engine: notrain
              n_trials: 5
              search_space:
                max_depth: {type: int, low: 2, high: 6}
              objective: {metric: pr_auc, direction: maximize}
            seed: 7
        """,
    )
    parsed = parse(demo_project, fake_registry)
    missing = [
        i
        for i in parsed.report.errors
        if "adapter 'no_such_engine_zzz' is not installed" in i.message
    ]
    assert missing and missing[0].field_path == "/tuning/engine"
    assert any("adapter 'notrain' provides no tuning engine" in m for m in error_messages(parsed))


def test_task_schema_spec_issues_are_reported(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "models/sliced.yml",
        """
        models:
          - name: m_sliced
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation:
              protocol: {split: temporal}
              metrics: [pr_auc]
              slices: [churned]
            seed: 7
        """,
    )
    parsed = parse(demo_project, fake_registry)
    issues = [i for i in parsed.report.errors if "slicing by the target column" in i.message]
    assert issues and issues[0].field_path == "/evaluation/slices/0"


def test_null_hyperparameter_is_deferred(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    model = demo_project / "models/churn_model.yml"
    model.write_text(
        model.read_text().replace("max_depth: 4", "max_depth: 4\n      scale_pos_weight:")
    )
    parsed = parse_project(demo_project, registry=fake_registry)  # no errors raised
    spec = parsed.models["model.demo.churn_model"].spec
    assert spec.hyperparameters["scale_pos_weight"] is None


# -- exposures --------------------------------------------------------------------


def test_exposure_error_branches(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "exposures.yml",
        """
        exposures:
          - name: e_invalid
          - name: dash
            type: dashboard
            owner: bi@example.com
            depends_on: ["ref('churn_model')"]
          - name: dash
            type: dashboard
            owner: bi@example.com
            depends_on: ["ref('churn_model')"]
          - name: dash_bare
            type: dashboard
            owner: bi@example.com
            depends_on: ["churn_model"]
          - name: dash_ghost
            type: dashboard
            owner: bi@example.com
            depends_on: ["ref('ghost')"]
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("required field 'type' is missing" in m for m in messages)
    assert any("duplicate exposure 'dash'" in m for m in messages)
    assert any("must be ref() calls, got 'churn_model'" in m for m in messages)
    assert any("exposure references unknown resource ref('ghost')" in m for m in messages)
    assert parsed.exposures["exposure.demo.dash"].depends_on == ["model.demo.churn_model"]


# -- scoring ----------------------------------------------------------------------


def test_scoring_capture_validation_and_maturity_branches(
    demo_project: Path, fake_registry: AdapterRegistry
) -> None:
    write(
        demo_project / "scoring/broken_scoring.yml",
        """
        scoring:
          - name: sc_var_missing
            description: "{{ var('missing_var_zzz') }}"
            owner: ds@example.com
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            output: {path: predictions/a, columns: [user_id]}
          - name: sc_invalid
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            output: {path: predictions/b, columns: [user_id]}
          - name: sc_dup
            owner: ds@example.com
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            output: {path: predictions/c, columns: [user_id]}
          - name: sc_dup
            owner: ds@example.com
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            output: {path: predictions/d, columns: [user_id]}
          - name: sc_bad_maturity
            owner: ds@example.com
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            ground_truth:
              label:
                source: source('lakehouse', 'subscribers')
                column: churned
              join_key: user_id
              maturity: "tomorrow"
              metrics: [pr_auc]
            output: {path: predictions/e, columns: [user_id]}
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("var 'missing_var_zzz' has no value at parse time" in m for m in messages)
    assert any("required field 'owner' is missing" in m for m in messages)
    assert any("duplicate scoring pipeline 'sc_dup'" in m for m in messages)
    maturity = [i for i in parsed.report.errors if "invalid window expression" in i.message]
    assert maturity and maturity[0].field_path == "/ground_truth/maturity"


def test_scoring_link_error_branches(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    register_unit_plugins(fake_registry)
    write(
        demo_project / "models/reggy_model.yml",
        """
        models:
          - name: reggy_model
            task: regression
            adapter: reggy
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
        """,
    )
    write(
        demo_project / "scoring/link_errors.yml",
        """
        scoring:
          - name: sc_bare_model
            owner: ds@example.com
            model: churn_model
            input:
              source: source('lakehouse', 'subscribers')
            output: {path: predictions/a, columns: [user_id]}
          - name: sc_extra_ref
            owner: ds@example.com
            description: "{{ ref('churn_training') }}"
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            output: {path: predictions/b, columns: [user_id]}
          - name: sc_no_schema
            owner: ds@example.com
            model: ref('reggy_model')
            input:
              source: source('lakehouse', 'subscribers')
            ground_truth:
              label:
                source: source('lakehouse', 'subscribers')
                column: churned
              join_key: user_id
              maturity: "14d"
              metrics: [pr_auc]
            output: {path: predictions/c, columns: [user_id]}
          - name: sc_bad_metric
            owner: ds@example.com
            model: ref('churn_model')
            input:
              source: source('lakehouse', 'subscribers')
            ground_truth:
              label:
                source: source('lakehouse', 'subscribers')
                column: churned
              join_key: user_id
              maturity: "14d"
              metrics: [nonexistent_metric_zzz]
            output: {path: predictions/d, columns: [user_id]}
          - name: sc_bare_source
            owner: ds@example.com
            model: ref('churn_model')
            input: {source: subscribers}
            output: {path: predictions/e, columns: [user_id]}
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("scoring 'model' must be a ref() call, got 'churn_model'" in m for m in messages)
    assert any("unexpected ref('churn_training') in scoring spec" in m for m in messages)
    assert any("nonexistent_metric_zzz" in m for m in messages)
    assert any("expected a source() reference, got 'subscribers'" in m for m in messages)
    # reggy has no task schema: ground-truth metrics resolve is skipped silently
    assert parsed.scoring["scoring.demo.sc_no_schema"].metric_specs == []


def test_scoring_inputs_form_parses(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "sources.yml",
        """
        sources:
          - name: lakehouse
            tables:
              - name: subscribers
                path: data/subscribers/*.parquet
              - name: extra_features
                path: data/subscribers/*.parquet
        """,
    )
    write(
        demo_project / "scoring/inputs_form.yml",
        """
        scoring:
          - name: sc_inputs
            owner: ds@example.com
            model: ref('churn_model')
            input:
              inputs:
                spine: source('lakehouse', 'subscribers')
                features: ["source('lakehouse', 'extra_features')"]
                join_key: user_id
            output: {path: predictions/a, columns: [user_id]}
        """,
    )
    parsed = parse_project(demo_project, registry=fake_registry)
    resource = parsed.scoring["scoring.demo.sc_inputs"]
    assert "source.demo.lakehouse.extra_features" in resource.depends_on
    assert "source.demo.lakehouse.subscribers" in resource.depends_on


# -- cross-resource linking ----------------------------------------------------------


def test_dataset_and_model_link_errors(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "datasets/link_errors.yml",
        """
        datasets:
          - name: ds_unknown_source
            source: source('nowhere', 'tbl')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
          - name: ds_with_ref
            description: "{{ ref('churn_training') }}"
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: temporal, time_column: t, train: "-30d:-7d", test: "-7d:now"}
        """,
    )
    write(
        demo_project / "models/link_errors.yml",
        """
        models:
          - name: m_source_call
            description: "{{ source('lakehouse', 'subscribers') }}"
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_bare_dataset
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: churn_training
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
          - name: m_extra_ref
            description: "{{ ref('something_else') }}"
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation: {protocol: {split: temporal}, metrics: [pr_auc]}
            seed: 7
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("unknown source ('nowhere', 'tbl')" in m for m in messages)
    assert any(
        "datasets cannot ref() other resources, got ref('churn_training')" in m for m in messages
    )
    assert any("models cannot use source() directly" in m for m in messages)
    assert any("model 'dataset' must be a ref() call, got 'churn_training'" in m for m in messages)
    assert any("unexpected ref('something_else') in model spec" in m for m in messages)


def test_model_test_window_validation(demo_project: Path, fake_registry: AdapterRegistry) -> None:
    write(
        demo_project / "datasets/rnd.yml",
        """
        datasets:
          - name: rnd_ds
            source: source('lakehouse', 'subscribers')
            label: {column: churned}
            split: {strategy: random, train: "0.8", test: "0.2", seed: 7}
            sample_key: user_id
        """,
    )
    write(
        demo_project / "models/windows.yml",
        """
        models:
          - name: m_rnd_window
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('rnd_ds')
            target: churned
            evaluation:
              protocol: {split: random, test_window: "-7d:now"}
              metrics: [pr_auc]
            seed: 7
          - name: m_bad_window
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation:
              protocol: {split: temporal, test_window: "bogus"}
              metrics: [pr_auc]
            seed: 7
          - name: m_not_subrange
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation:
              protocol: {split: temporal, test_window: "-60d:now"}
              metrics: [pr_auc]
            seed: 7
          - name: m_good_window
            task: binary_classification
            adapter: fake
            owner: ds@example.com
            dataset: ref('churn_training')
            target: churned
            evaluation:
              protocol: {split: temporal, test_window: "-14d:now"}
              metrics: [pr_auc]
            seed: 7
        """,
    )
    parsed = parse(demo_project, fake_registry)
    messages = error_messages(parsed)
    assert any("test_window requires a temporal split" in m for m in messages)
    assert any("invalid window expression 'bogus'" in m for m in messages)
    assert any("test_window '-60d:now' must resolve to a sub-range" in m for m in messages)
    assert not any("m_good_window" in (i.resource or "") for i in parsed.report.errors)


def test_dependency_cycle_is_reported_by_graph_builder() -> None:
    """Spec layering makes cycles impossible via parse; the guard is unit-tested."""
    from mbt.contracts import MetricSpec
    from mbt.parsing.errors import ParseReport
    from mbt.parsing.project_parser import ParsedResource, _build_project_graph

    def resource(uid: str, dep: str) -> ParsedResource:
        return ParsedResource(
            unique_id=uid,
            resource_type="dataset",
            name=uid.rsplit(".", 1)[-1],
            path="x.yml",
            spec=MetricSpec(name="m"),
            raw={},
            depends_on=[dep],
        )

    datasets = {
        "dataset.p.a": resource("dataset.p.a", "dataset.p.b"),
        "dataset.p.b": resource("dataset.p.b", "dataset.p.a"),
    }
    report = ParseReport()
    _build_project_graph({}, datasets, {}, {}, {}, report)
    cycle = [i for i in report.errors if "dependency cycle detected" in i.message]
    assert cycle and "dataset.p.a" in cycle[0].message
