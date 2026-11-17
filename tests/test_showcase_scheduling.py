"""The scheduling/CD half of P4 (SHOW-11 strong form + SHOW-13 routing).

Airflow (LocalExecutor + postgres) runs the DAGs that git-sync reconciles
out of the Gitea `deploy` repo, drawn by mbt-airflow (ADR-33, ADR-35): a task
per step of each node (a build node is several steps; a score or monitor node
is one), from the manifest prod-build published beside the image pin. Every
task runs mbt inside the deployable unit pinned by digest in images.env
(ADR-34). This module proves:

- the baseline retrain DAG builds on the PROD target from a scheduled, pinned
  unit: cluster pushdown from a container Airflow launched (the deterministic
  xgboost workhorse - sparkling-on-cluster is SHOW-04's own module, kept
  out of scheduled paths per DESIGN's flake-isolation rule), as the dataset's
  materialize and checks tasks and the model's train, gate and register tasks,
  each reading what the one before left;
- SHOW-11, the ADR-20 inversion end to end: two score DAG runs straddling a
  promotion serve different champions while the deploy repo HEAD and the
  pinned image digest stay byte-identical - promotion is a registry event,
  never a deploy;
- SHOW-13 at the scheduler: a realized-gate breach (exit 2) fails the task
  on try 1 with NO retry (quality verdicts are deterministic; the owner is
  notified, not on-call), while a hard error (exit 1) consumes a retry
  before failing.
"""

import pytest
from showcase_utils import ANCHOR, SHOWCASE_MARKS

pytestmark = SHOWCASE_MARKS

DAGS = ("mbt_retrain", "mbt_retrain_baseline", "mbt_score", "mbt_monitor")
_state: dict = {}


def _client(stack):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=stack.mlflow_url())


def _predictions_root(stack):
    return stack.workspace / "predictions" / "retention_scores"


def _sidecars(stack) -> list[dict]:
    """All prediction sidecars, oldest first by scored mtime."""
    import json

    root = _predictions_root(stack)
    if not root.is_dir():
        return []
    docs = []
    for run_dir in root.iterdir():
        sidecar = run_dir / "predictions.json"
        if sidecar.exists():
            docs.append((sidecar.stat().st_mtime, json.loads(sidecar.read_text())))
    return [doc for _, doc in sorted(docs, key=lambda pair: pair[0])]


@pytest.fixture(scope="module")
def sched(showcase_ci):
    ci = showcase_ci
    ci.ensure_seeded()

    # git-sync + the DAG processor need a few cycles after the deploy repo
    # gained its digest pin; every DAG must be registered and live.
    for dag_id in DAGS:
        ci.wait_dag(dag_id)

    # The production champion must exist (it does after the lifecycle
    # module; a standalone run promotes the freshest gate-stamped staging
    # version - the CI bootstrap registered it).
    client = _client(ci.stack)
    try:
        client.get_model_version_by_alias("churn_automl", "production")
    except Exception:
        ci.stack.mbt("promote", "--model", "churn_automl", "--to", "production", timeout=300)
    return ci


def test_retrain_dag_builds_on_cluster_from_pinned_unit(sched) -> None:
    """The retrain path: scheduler -> pinned unit -> prod target."""
    ci = sched
    client = _client(ci.stack)
    before = len(client.search_model_versions("name='churn_baseline_xgb'"))
    master_before = ci.stack.http_json(
        f"http://localhost:{ci.stack.ports['SHOWCASE_SPARK_UI_PORT']}/json/"
    )
    apps_before = len(master_before.get("completedapps", [])) + len(
        master_before.get("activeapps", [])
    )

    run_id = ci.trigger_dag("mbt_retrain_baseline")
    assert ci.wait_dag_run("mbt_retrain_baseline", run_id) == "success"

    # A node is a group of steps (ADR-35), each its own container: the lake's
    # anchor was asked for once by the compile task, the dataset was built and
    # then checked, and the model was trained, judged and registered - the
    # registry write in a task of its own, so a blip there never retrains.
    tasks = ci.task_instances("mbt_retrain_baseline", run_id)
    expected = {
        "mbt_compile",
        "churn_training.materialize",
        "churn_training.checks",
        "churn_baseline_xgb.train",
        "churn_baseline_xgb.gate",
        "churn_baseline_xgb.register",
    }
    assert set(tasks) == expected, tasks
    assert {task["state"] for task in tasks.values()} == {"success"}, tasks
    assert all(task["try_number"] == 1 for task in tasks.values()), tasks

    # A new version registered from the scheduled run, trained via cluster
    # pushdown (the master saw new applications from the unit container).
    versions = client.search_model_versions("name='churn_baseline_xgb'")
    assert len(versions) == before + 1
    assert int(client.get_model_version_by_alias("churn_baseline_xgb", "staging").version) == max(
        int(v.version) for v in versions
    )
    master_after = ci.stack.http_json(
        f"http://localhost:{ci.stack.ports['SHOWCASE_SPARK_UI_PORT']}/json/"
    )
    apps_after = len(master_after.get("completedapps", [])) + len(
        master_after.get("activeapps", [])
    )
    assert apps_after > apps_before, (apps_before, apps_after)


def test_score_dags_straddling_promotion_flip_champion_zero_redeploy(sched) -> None:
    """SHOW-11: promotion changes the NEXT scheduled run; CD sees nothing."""
    ci = sched
    client = _client(ci.stack)
    served_before = int(client.get_model_version_by_alias("churn_automl", "production").version)

    run_id = ci.trigger_dag("mbt_score")
    assert ci.wait_dag_run("mbt_score", run_id) == "success"
    sidecar = _sidecars(ci.stack)[-1]
    assert str(sidecar["model_version"]) == str(served_before), sidecar

    deploy_head = ci.deploy_commits()[0]["sha"]
    image = ci.images_env()["IMAGE"]

    # Promote a DIFFERENT gate-stamped version (pinned, a pure registry
    # event - re-promoting history is exactly how a rollback looks). A
    # standalone module run has only the seeded version; mint a challenger
    # deterministically (dev target, same backend as the champion, so the
    # paired champion gate sees identical predictions and passes).
    def _candidates() -> list:
        return [
            int(v.version)
            for v in client.search_model_versions("name='churn_automl'")
            if int(v.version) != served_before and v.tags.get("mbt.gates_passed") == "true"
        ]

    candidates = _candidates()
    if not candidates:
        ci.stack.mbt("build", "--target", "dev", "--select", "churn_automl", "--anchor", ANCHOR)
        candidates = _candidates()
    assert candidates, "no other gate-stamped churn_automl version to promote"
    promoted = max(candidates)
    ci.stack.mbt(
        "promote",
        "--model",
        "churn_automl",
        "--to",
        "production",
        "--version",
        str(promoted),
        timeout=300,
    )

    run_id = ci.trigger_dag("mbt_score")
    assert ci.wait_dag_run("mbt_score", run_id) == "success"
    sidecar = _sidecars(ci.stack)[-1]
    assert str(sidecar["model_version"]) == str(promoted), sidecar

    # The ADR-20 inversion, asserted: zero redeploy happened.
    assert ci.deploy_commits()[0]["sha"] == deploy_head
    assert ci.images_env()["IMAGE"] == image
    _state["scored_versions"] = (served_before, promoted)


def test_monitor_dag_routes_exit_codes(sched) -> None:
    """SHOW-13 at the scheduler: 2 = fail fast to the owner; 1 = retry."""
    ci = sched
    # The scored cohort's outcomes land in the lake table, so the monitor has
    # something to grade; put the table back as seeded afterwards.
    ci.stack.panel("land-outcomes")
    try:
        _monitor_dag_routes_exit_codes(ci)
    finally:
        ci.stack.panel("reset")


def _monitor_dag_routes_exit_codes(ci) -> None:
    # Quality verdict: an impossible realized floor breaches the gate on
    # the freshly scored (unevaluated) runs -> mbt exits 2 -> the task
    # fails on try 1, retries NOT consumed (AirflowFailException).
    run_id = ci.trigger_dag("mbt_monitor", {"vars": "pr_auc_floor: 0.99"})
    assert ci.wait_dag_run("mbt_monitor", run_id) == "failed"
    task = ci.task_instances("mbt_monitor", run_id)["retention_scoring"]
    assert task["state"] == "failed"
    assert task["try_number"] == 1, task

    # Exactly-once survives the scheduler: the breach evaluated those runs,
    # so a normal re-run finds nothing left and succeeds.
    run_id = ci.trigger_dag("mbt_monitor")
    assert ci.wait_dag_run("mbt_monitor", run_id) == "success"

    # Hard error: a pinned anchor that is not a date exits 1 at the compile
    # task -> Airflow RETRIES first (try_number 2), then fails toward on-call,
    # and the scoring task behind it never starts.
    run_id = ci.trigger_dag("mbt_monitor", {"anchor": "not-a-date"})
    assert ci.wait_dag_run("mbt_monitor", run_id) == "failed"
    tasks = ci.task_instances("mbt_monitor", run_id)
    assert tasks["mbt_compile"]["state"] == "failed"
    assert tasks["mbt_compile"]["try_number"] == 2, tasks["mbt_compile"]
    assert tasks["retention_scoring"]["state"] == "upstream_failed", tasks["retention_scoring"]

    # And the monitored runs really were this module's two score runs.
    monitored = {str(v) for v in _state.get("scored_versions", ())}
    sidecar_versions = {str(doc["model_version"]) for doc in _sidecars(ci.stack)}
    assert monitored <= sidecar_versions
