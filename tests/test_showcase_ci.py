"""The showcase Woodpecker CI loop, end to end (SHOW-05/06/07 + SHOW-10's
authorization edge).

Uses the session-scoped `showcase_ci` harness (gitea + woodpecker + zot +
webhook-sink, headlessly seeded by examples/showcase/scripts/ci_bootstrap.py)
and drives the committed .woodpecker/ pipelines exactly as a user would:
git pushes and PRs against Gitea, assertions through Gitea's API (PR
comments, the mbt-state branch), Woodpecker's API (pipeline status), the
shared MLflow registry, and the webhook-sink alert recorder.

Deliberate order within this module (each test builds on the previous):

1. Bootstrap: first push to main -> prod-build honors fetch_state exit 3,
   trains everything, registers to the SHARED registry, publishes the
   mbt-state baseline (SHOW-06).
2. No-change PR -> "Nothing modified" comment; merge -> baseline
   republished with identical nodes, registry untouched (SHOW-06).
3. One-gate-edit PR -> slim CI: the comment shows exactly the edited model
   as modified (config) plus its downstream scoring node (upstream) - and
   NO dataset churn across fresh clones (URI snapshot stability, SHOW-05).
4. Merge -> state economy: only the modified model retrains (SHOW-05/06).
5. Impossible-gate PR -> quality failure: pipeline fails with mbt's exit 2
   classified as such (never 1), the comment shows gate_failed, the shared
   registry is untouched, and webhook-sink holds exactly one
   owner-classified alert (SHOW-07).
6. Protected promotion: branch protection + CODEOWNERS gate promotions.yml,
   an unauthorized direct push is rejected, and the owner-approved merge
   runs the promote pipeline, moving the production alias (SHOW-10).
"""

import pytest
from showcase_utils import (
    DS_USER,
    GITEA_USER,
    ORG,
    REPO,
    SHOWCASE_MARKS,
    run_git,
)

pytestmark = SHOWCASE_MARKS

AUTOML = "model.churn_lake.churn_automl"
SCORING = "scoring.churn_lake.retention_scoring"
GATE_LINE = "threshold: \"{{ var('pr_auc_floor') }}\""
_state: dict = {}


@pytest.fixture(scope="module")
def ci(showcase_ci):
    return showcase_ci


def _mlflow_versions(stack, name: str) -> list:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=stack.mlflow_url())
    return client.search_model_versions(f"name='{name}'")


def test_bootstrap_full_build_publishes_baseline(ci) -> None:
    """SHOW-06: the first-ever prod-build honors fetch_state exit 3."""
    assert ci.ensure_seeded() is True, "expected a virgin forge (this module runs first)"

    # Bootstrap = full build: both models registered in the SHARED registry
    # with staging aliases and the gate stamp `mbt promote` verifies.
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=ci.stack.mlflow_url())
    for model in ("churn_automl", "churn_baseline_xgb"):
        version = client.get_model_version_by_alias(model, "staging")
        assert int(version.version) == 1, (model, version.version)
        assert version.tags.get("mbt.gates_passed") == "true"

    # The baseline landed on the mbt-state branch...
    manifest = ci.state_manifest()
    assert AUTOML in manifest["nodes"]
    assert len(ci.state_commits()) == 1

    # ...and the first deployable unit was baked and pinned (P3): the deploy
    # repo points at a digest that zot can serve.
    conf = ci.images_env()
    assert "@sha256:" in conf["IMAGE"], conf
    _state["bootstrap_image"] = conf["IMAGE"]


def test_no_change_pr_and_republish(ci) -> None:
    """SHOW-06: a no-change build trains nothing; the baseline republishes,
    and no new deployable unit is baked (the digest pin is untouched)."""
    clone = ci.fresh_clone("nochange")
    run_git(clone, "checkout", "-b", "docs-tweak")
    readme = clone / "README.md"
    readme.write_text(readme.read_text() + "\nDocs-only change.\n")
    run_git(clone, "add", "README.md")
    run_git(clone, "commit", "-m", "docs: no pipeline-relevant change")
    run_git(clone, "push", "origin", "docs-tweak")

    pr = ci.gitea_api(
        "POST",
        f"/repos/{ORG}/{REPO}/pulls",
        {"title": "docs tweak", "head": "docs-tweak", "base": "main"},
    )
    pipeline = ci.wait_pipeline("pull_request")
    assert pipeline["status"] == "success", pipeline

    comment = ci.pr_comment(pr["number"])
    assert "Nothing modified - no retraining needed." in comment
    assert "- modified:" not in comment

    before = {m: len(_mlflow_versions(ci.stack, m)) for m in ("churn_automl", "churn_baseline_xgb")}
    baseline_nodes = ci.state_manifest()["nodes"]
    deploy_head = ci.deploy_commits()[0]["sha"]

    ci.merge_pr(pr["number"])
    pipeline = ci.wait_pipeline("push")
    assert pipeline["status"] == "success", pipeline

    # Republished (a new state commit) yet inert: identical nodes, zero
    # registry churn, and the deploy repo HEAD (the digest pin) untouched -
    # nothing retrained means nothing to deploy.
    assert len(ci.state_commits()) == 2
    assert ci.state_manifest()["nodes"] == baseline_nodes
    after = {m: len(_mlflow_versions(ci.stack, m)) for m in ("churn_automl", "churn_baseline_xgb")}
    assert after == before, (before, after)
    assert ci.deploy_commits()[0]["sha"] == deploy_head


def test_slim_pr_shows_exactly_the_edited_node(ci) -> None:
    """SHOW-05: one gate edit -> one modified model (config), no dataset churn."""
    clone = ci.fresh_clone("gate-edit")
    run_git(clone, "checkout", "-b", "raise-gate")
    spec = clone / "models" / "churn_automl.yml"
    text = spec.read_text()
    assert GATE_LINE in text, f"gate line moved; update the test: {text}"
    spec.write_text(text.replace(GATE_LINE, "threshold: 0.31"))
    run_git(clone, "add", "models/churn_automl.yml")
    run_git(clone, "commit", "-m", "raise churn_automl pr_auc floor to 0.31")
    run_git(clone, "push", "origin", "raise-gate")

    pr = ci.gitea_api(
        "POST",
        f"/repos/{ORG}/{REPO}/pulls",
        {"title": "raise automl gate", "head": "raise-gate", "base": "main"},
    )
    pipeline = ci.wait_pipeline("pull_request")
    assert pipeline["status"] == "success", pipeline

    comment = ci.pr_comment(pr["number"])
    # Exactly the edited model is modified (config); its scoring node is
    # flagged as downstream lineage (upstream), and NOTHING else - fresh
    # clones must not churn datasets (URI snapshot stability).
    modified = [line for line in comment.splitlines() if line.startswith("- modified:")]
    assert f"- modified: `{AUTOML}` (config)" in modified, comment
    assert f"- modified: `{SCORING}` (upstream)" in modified, comment
    assert len(modified) == 2, modified
    assert "dataset." not in "".join(modified), modified
    assert "- added:" not in comment

    # The slim build trained the modified model and its gates passed...
    assert f"| `{AUTOML}` | success |" in comment
    assert "**FAIL**" not in comment
    # ...against the PR-scoped registry: the SHARED registry is untouched.
    assert len(_mlflow_versions(ci.stack, "churn_automl")) == 1

    _state["gate_pr"] = pr["number"]


def test_merge_economy_retrains_only_modified(ci) -> None:
    """SHOW-05/06: merging the gate edit retrains ONLY the edited model,
    and a real retrain means a new deployable unit gets pinned."""
    deploy_head = ci.deploy_commits()[0]["sha"]
    ci.merge_pr(_state["gate_pr"])
    pipeline = ci.wait_pipeline("push")
    assert pipeline["status"] == "success", pipeline

    # Economy: churn_automl got version 2 (and the staging alias moved);
    # the untouched baseline model did NOT retrain.
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=ci.stack.mlflow_url())
    assert len(_mlflow_versions(ci.stack, "churn_automl")) == 2
    assert int(client.get_model_version_by_alias("churn_automl", "staging").version) == 2
    assert len(_mlflow_versions(ci.stack, "churn_baseline_xgb")) == 1

    assert len(ci.state_commits()) == 3

    # This merge retrained -> the unit re-baked and the pin moved (P3).
    assert ci.deploy_commits()[0]["sha"] != deploy_head
    conf = ci.images_env()
    assert "@sha256:" in conf["IMAGE"]
    assert conf["IMAGE"] != _state["bootstrap_image"]


def test_gate_failure_is_exit_2_owner_alert(ci) -> None:
    """SHOW-07: a failing gate is a quality verdict (2), never a hard error."""
    ci.reset_alerts()

    clone = ci.fresh_clone("gate-fail")
    run_git(clone, "checkout", "-b", "impossible-gate")
    spec = clone / "models" / "churn_automl.yml"
    spec.write_text(spec.read_text().replace("threshold: 0.31", "threshold: 0.99"))
    run_git(clone, "add", "models/churn_automl.yml")
    run_git(clone, "commit", "-m", "set an impossible pr_auc floor")
    run_git(clone, "push", "origin", "impossible-gate")

    pr = ci.gitea_api(
        "POST",
        f"/repos/{ORG}/{REPO}/pulls",
        {"title": "impossible gate", "head": "impossible-gate", "base": "main"},
    )
    pipeline = ci.wait_pipeline("pull_request")
    assert pipeline["status"] == "failure", pipeline

    # The comment still posted (the step runs on failure) and shows the
    # quality verdict, not a crash.
    comment = ci.pr_comment(pr["number"])
    assert f"| `{AUTOML}` | gate_failed |" in comment
    assert "**FAIL**" in comment
    assert "node(s) failed - registration blocked" in comment

    # run_mbt.sh classified exit 2 as a quality failure and notified the
    # OWNER (exit 1 would page on-call instead) - exactly once.
    alerts = [a for a in ci.alerts() if a["path"] == "/alert"]
    assert len(alerts) == 1, alerts
    body = alerts[0]["body"]
    assert body["class"] == "quality-failure", body
    assert body["notify"] == "owner", body
    assert body["owner"] == "growth-ds@example.com", body
    assert AUTOML in body["failed_nodes"], body

    # No registry churn from a red PR.
    assert len(_mlflow_versions(ci.stack, "churn_automl")) == 2


def test_protected_promotion_flow(ci) -> None:
    """SHOW-10 (CI side): promotions.yml is gated by branch protection +
    CODEOWNERS; the owner-approved merge runs the promote pipeline."""
    # CODEOWNERS routes promotions.yml reviews to mbtops (pushed by mbtops
    # BEFORE protection tightens main).
    clone = ci.fresh_clone("codeowners")
    (clone / "CODEOWNERS").write_text("promotions.yml @mbtops\n")
    run_git(clone, "add", "CODEOWNERS")
    run_git(clone, "commit", "-m", "ownership: promotions.yml belongs to mbtops")
    run_git(clone, "push", "origin", "main")
    assert ci.wait_pipeline("push")["status"] == "success"

    # Protection: only mbtops may push main directly; merges need one
    # approval from the owners' whitelist. (Gitea API quirk: the approvals
    # whitelist field is singular `approvals_whitelist_username` while the
    # push one is plural - a misspelling is silently dropped, leaving an
    # empty whitelist that makes every review unofficial.)
    protection = ci.gitea_api(
        "POST",
        f"/repos/{ORG}/{REPO}/branch_protections",
        {
            "branch_name": "main",
            "enable_push": True,
            "enable_push_whitelist": True,
            "push_whitelist_usernames": [GITEA_USER],
            "required_approvals": 1,
            "enable_approvals_whitelist": True,
            "approvals_whitelist_username": [GITEA_USER],
        },
    )
    assert protection["approvals_whitelist_username"] == [GITEA_USER], protection

    # The DS persona pins the v2 champion for production...
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=ci.stack.mlflow_url())
    version = int(client.get_model_version_by_alias("churn_automl", "staging").version)
    promotion = (
        f"promotions:\n  - model: churn_automl\n    to: production\n    version: '{version}'\n"
    )

    ds_clone = ci.fresh_clone("promote", user=DS_USER, token=ci.ds_token)
    (ds_clone / "promotions.yml").write_text(promotion)
    run_git(ds_clone, "add", "promotions.yml")
    run_git(ds_clone, "commit", "-m", f"promote churn_automl v{version} to production")

    # ...but their direct push to main bounces off branch protection.
    rejected = run_git(ds_clone, "push", "origin", "main", expect_ok=False)
    assert rejected.returncode != 0
    assert "protected" in (rejected.stdout + rejected.stderr).lower(), rejected.stderr

    # The PR route: mbtds proposes, mbtops (the code owner) approves.
    run_git(ds_clone, "checkout", "-b", "promote-automl")
    run_git(ds_clone, "push", "origin", "promote-automl")
    pr = ci.gitea_api(
        "POST",
        f"/repos/{ORG}/{REPO}/pulls",
        {"title": f"promote churn_automl v{version}", "head": "promote-automl", "base": "main"},
        token=ci.ds_token,
    )
    assert ci.wait_pipeline("pull_request")["status"] == "success"

    # Unapproved merge is refused; the owner's approval unlocks it.
    import requests

    refused = requests.post(
        f"{ci.stack.gitea_url()}/api/v1/repos/{ORG}/{REPO}/pulls/{pr['number']}/merge",
        json={"Do": "merge"},
        headers={"Authorization": f"token {ci.ds_token}"},
        timeout=30,
    )
    assert refused.status_code == 405, (refused.status_code, refused.text[:300])
    ci.gitea_api("POST", f"/repos/{ORG}/{REPO}/pulls/{pr['number']}/reviews", {"event": "APPROVED"})
    ci.merge_pr(pr["number"])

    # The merge triggers prod-build AND the path-filtered promote workflow;
    # promotion is a registry event: production alias moves, deploy repo
    # HEAD and the pinned digest stay byte-identical (ADR-20).
    deploy_head = ci.deploy_commits()[0]["sha"]
    image = ci.images_env()["IMAGE"]
    assert ci.wait_pipeline("push")["status"] == "success"
    champion = client.get_model_version_by_alias("churn_automl", "production")
    assert int(champion.version) == version
    assert ci.deploy_commits()[0]["sha"] == deploy_head
    assert ci.images_env()["IMAGE"] == image

    # Leave the registry as the later modules expect it: the lifecycle
    # module's H2O builds must meet an empty production stage (a champion
    # trained by a different backend would put their champion gates inside
    # H2O's 0.02 determinism tier - nondeterministic by construction).
    client.delete_registered_model_alias("churn_automl", "production")
