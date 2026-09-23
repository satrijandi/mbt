"""The parser's invariants as a registry (A-3), and the compile-phase hole.

Verifying one rule used to mean editing fixture YAML by string replacement and
running a full parse. With the rules in a list, each is a two-spec table test
and "every rule runs" is one test over the list.
"""

from pathlib import Path

import pytest
from core_helpers import write

from mbt.parsing import parse_project
from mbt.parsing.rules import BOTH_PHASES, RULES, Rule, run_rules

ANCHOR = "2026-07-01T00:00:00Z"


# -- the registry itself ----------------------------------------------------


def test_every_rule_is_uniquely_named() -> None:
    names = [rule.name for rule in RULES]
    assert len(names) == len(set(names)), names


def test_every_rule_is_a_rule_with_a_known_target_kind() -> None:
    for rule in RULES:
        assert isinstance(rule, Rule)
        assert rule.applies_to in ("dataset", "model", "scoring")
        assert callable(rule.check)


def test_rules_run_in_both_phases_unless_they_say_otherwise() -> None:
    """The default is BOTH, which is what closes the compile-phase hole.

    A rule that opts out has to say so in the registry, where a reader sees it -
    rather than the opt-out being the silent default it effectively was when
    compile re-ran exactly one of the nineteen rules.
    """
    both = [rule.name for rule in RULES if rule.phases == BOTH_PHASES]
    parse_only = [rule.name for rule in RULES if rule.phases == ("parse",)]
    assert len(both) + len(parse_only) == len(RULES)
    # Only the unresolved-reference SYNTAX rules may be parse-only: the resolve
    # phase rewrites source('a','b') into a unique_id, so they cannot hold after.
    assert sorted(parse_only) == ["dataset.source_syntax", "scoring.source_syntax"]


def test_run_rules_honours_the_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    from mbt.parsing.errors import ParseReport
    from mbt.parsing.rules import ResolvedTarget, RuleContext

    seen: list[str] = []
    rules = (
        Rule("both", "dataset", lambda t, c, r: seen.append("both")),
        Rule("parse_only", "dataset", lambda t, c, r: seen.append("parse_only"), phases=("parse",)),
    )
    monkeypatch.setattr("mbt.parsing.rules.RULES", rules)
    target = ResolvedTarget(unique_id="dataset.p.d", name="d", path="d.yml", spec=object())

    def _ctx(phase: str) -> RuleContext:
        return RuleContext(
            project_name="p",
            datasets={"dataset.p.d": target},
            models={},
            scoring={},
            depends_on={},
            registry=None,  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
        )

    run_rules(_ctx("parse"), ParseReport())
    assert seen == ["both", "parse_only"]
    seen.clear()
    run_rules(_ctx("compile"), ParseReport())
    assert seen == ["both"]


# -- the compile-phase hole -------------------------------------------------


def _project(root: Path, *, target_var: str, test_window: str) -> Path:
    """A project whose model reads ``evaluation.protocol.test_window`` from a
    TARGET var, so the value only exists after resolve-rendering."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "data").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "user_id": list(range(80)),
                "ts": [f"2026-0{1 + i % 6}-10T00:00:00" for i in range(80)],
                "churned": [i % 2 for i in range(80)],
            }
        ),
        root / "data" / "rows.parquet",
    )
    # A safe project-level default, which the TARGET below overrides. This is
    # exactly the shape ADR-5's Nuance describes: parse can only see the
    # default, compile sees the target's value - so a rule that runs at parse
    # and not at compile never examines the value execution uses.
    write(
        root / "mbt_project.yml",
        "name: hole_demo\nversion: '0.1.0'\nvars:\n  window_expr: '-30d:now'\n",
    )
    write(
        root / "profiles.yml",
        f"""
hole_demo:
  target: dev
  outputs:
    dev:
      data: {{adapter: local, config: {{root: {root}}}}}
      tracking: {{adapter: fake, config: {{root: {root}/target/ft}}}}
      registry: {{adapter: fake, config: {{root: {root}/target/fr}}}}
      compute: {{adapter: local}}
      artifact_store: file://{root}/target/artifacts
      vars:
        {target_var}
""",
    )
    write(
        root / "sources.yml",
        "sources:\n  - name: lake\n    tables:\n      - name: rows\n"
        "        path: data/*.parquet\n        format: parquet\n",
    )
    write(
        root / "datasets" / "panel.yml",
        """
datasets:
  - name: panel
    source: source('lake', 'rows')
    sample_key: [user_id]
    label:
      column: churned
    split:
      strategy: temporal
      time_column: ts
      train: '-180d:-60d'
      test: '-60d:now'
""",
    )
    write(
        root / "models" / "clf.yml",
        f"""
models:
  - name: clf
    dataset: ref('panel')
    adapter: xgboost
    task: binary_classification
    owner: ds@example.com
    target: churned
    seed: 42
    evaluation:
      metrics: [roc_auc]
      protocol:
        split: temporal
        test_window: "{test_window}"
""",
    )
    return root


def test_a_target_var_moving_test_window_out_of_range_is_now_caught(tmp_path: Path) -> None:
    """A-3's compile-phase hole, as a regression test.

    ``evaluation.protocol.test_window`` must resolve to a sub-range of the
    dataset's test window. That rule ran at parse, where ``{{ var(...) }}`` is
    still unresolved - and compile, which CAN see the value, re-ran exactly one
    of the nineteen rules and not this one. So a target var that moved the
    window past the dataset's reached execution unchecked.
    """
    from mbt.compile.compiler import compile_project
    from mbt.config.profiles import load_profiles
    from mbt.exceptions import ConfigError

    root = _project(
        tmp_path / "proj",
        target_var='window_expr: "-400d:-300d"',  # nowhere near the dataset's test window
        test_window="{{ var('window_expr') }}",
    )
    parsed = parse_project(root)
    profiles = load_profiles("hole_demo", root)
    with pytest.raises(ConfigError, match="must resolve to a sub-range"):
        compile_project(parsed, profiles)


def test_a_target_var_inside_the_dataset_window_still_compiles(tmp_path: Path) -> None:
    """The other half of the table: a legal value must not be flagged."""
    from mbt.compile.compiler import compile_project
    from mbt.config.profiles import load_profiles

    root = _project(
        tmp_path / "proj",
        target_var='window_expr: "-30d:now"',  # inside the dataset's -60d:now test window
        test_window="{{ var('window_expr') }}",
    )
    parsed = parse_project(root)
    manifest = compile_project(parsed, load_profiles("hole_demo", root))
    assert "model.hole_demo.clf" in manifest.nodes


# -- D-4 --------------------------------------------------------------------


def _rebalanced_model(root: Path, *, calibration: str | None, metrics: str) -> Path:
    _project(root, target_var="unused: 1", test_window="-30d:now")
    calibration_line = f"    calibration: {calibration}\n" if calibration else ""
    write(
        root / "models" / "clf.yml",
        f"""
models:
  - name: clf
    dataset: ref('panel')
    adapter: xgboost
    task: binary_classification
    owner: ds@example.com
    target: churned
    seed: 42
{calibration_line}    hyperparameters:
      scale_pos_weight: '{{{{ auto }}}}'
    evaluation:
      protocol: {{split: temporal}}
      metrics: {metrics}
""",
    )
    return root


def test_auto_rebalancing_plus_a_calibration_metric_with_no_calibrator_warns(
    tmp_path: Path,
) -> None:
    """D-4. R2-8 was this exact combination and was closed twice - by building
    calibration, then by fixing the demo fixture. Neither added a guard."""
    root = _rebalanced_model(tmp_path / "proj", calibration=None, metrics="[roc_auc, brier]")
    parsed = parse_project(root)
    warnings = [w.message for w in parsed.report.warnings]
    assert any("measures calibration and no 'calibration:' is set" in w for w in warnings), warnings


def test_a_calibrator_makes_the_same_spec_quiet(tmp_path: Path) -> None:
    root = _rebalanced_model(tmp_path / "proj", calibration="isotonic", metrics="[roc_auc, brier]")
    parsed = parse_project(root)
    assert not [w for w in parsed.report.warnings if "measures calibration" in w.message]


def test_rebalancing_without_a_calibration_metric_is_quiet(tmp_path: Path) -> None:
    """Rebalancing for ranking is a legitimate thing to do; only the
    combination with an uncalibrated calibration METRIC is meaningless."""
    root = _rebalanced_model(tmp_path / "proj", calibration=None, metrics="[roc_auc, pr_auc]")
    parsed = parse_project(root)
    assert not [w for w in parsed.report.warnings if "measures calibration" in w.message]


def test_rule_context_finds_a_models_dataset_and_a_pipelines_model() -> None:
    """Rules read neighbours through the LINKED graph, which is what lets the
    same rule run at parse and at compile (A-3)."""
    from mbt.parsing.rules import ResolvedTarget, RuleContext

    dataset = ResolvedTarget("dataset.p.d", "d", "d.yml", object())
    model = ResolvedTarget("model.p.m", "m", "m.yml", object())
    scoring = ResolvedTarget("scoring.p.s", "s", "s.yml", object())
    ctx = RuleContext(
        project_name="p",
        datasets={dataset.unique_id: dataset},
        models={model.unique_id: model},
        scoring={scoring.unique_id: scoring},
        depends_on={model.unique_id: [dataset.unique_id], scoring.unique_id: [model.unique_id]},
        registry=None,  # type: ignore[arg-type]
    )
    assert ctx.dataset_of(model) is dataset
    assert ctx.model_of(scoring) is model
    assert ctx.dataset_by_name() == {"d": dataset}
    # an unlinked resource has no neighbour, and a rule must cope
    orphan = ResolvedTarget("model.p.orphan", "orphan", "o.yml", object())
    assert ctx.dataset_of(orphan) is None
    assert ctx.model_of(orphan) is None


def test_a_jinja_scale_pos_weight_is_not_statically_checkable() -> None:
    """D-4 only fires on values it can actually read: an unresolved expression
    is not a number, and guessing would produce a false warning."""
    from mbt.parsing.rules import _is_large_weight

    assert _is_large_weight(4.0) is True
    assert _is_large_weight("4.0") is True
    assert _is_large_weight(1.0) is False
    assert _is_large_weight("{{ var('w') }}") is False
    assert _is_large_weight(None) is False


def test_a_mild_scale_pos_weight_is_not_a_rebalance(tmp_path: Path) -> None:
    """D-4 fires on a deliberate rebalance, not on any weight at all: the auto
    value for a 20% positive rate is 4.0, so a nudge below the bar is quiet."""
    root = _rebalanced_model(tmp_path / "proj", calibration=None, metrics="[roc_auc, brier]")
    spec = root / "models" / "clf.yml"
    spec.write_text(
        spec.read_text().replace("scale_pos_weight: '{{ auto }}'", "scale_pos_weight: 1.2")
    )
    parsed = parse_project(root)
    assert not [w for w in parsed.report.warnings if "measures calibration" in w.message]
