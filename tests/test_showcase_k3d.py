"""The optional k3d + ArgoCD CD-fidelity profile (SHOW-16, DESIGN.md P6).

Local-only and DOUBLY opt-in (MBT_LIVE_SHOWCASE=1 AND MBT_LIVE_SHOWCASE_K3D=1;
the nightly job sets only the former): a k3d cluster attached to the compose
network, ArgoCD core syncing the Gitea `deploy` repo's k8s/ dir, zot marked
insecure via registries.yaml. Proves the literal-fidelity CD claims:

- ArgoCD renders the deploy repo into the mbt namespace (CronJob pinned to
  the baked digest);
- an insecure-HTTP pull from zot works inside the cluster (a pod runs
  `mbt --version` from the pinned unit);
- a digest bump in the deploy repo rolls the CronJob spec;
- selfHeal recreates a deleted CronJob without any human action.
"""

import os
import shutil
import subprocess

import pytest
from showcase_utils import SHOWCASE_MARKS, run_git

pytestmark = [
    *SHOWCASE_MARKS,
    pytest.mark.skipif(
        os.environ.get("MBT_LIVE_SHOWCASE_K3D") != "1",
        reason="k3d/ArgoCD fidelity profile is separately opt-in: set MBT_LIVE_SHOWCASE_K3D=1",
    ),
]

ARGOCD_CORE = (
    "https://raw.githubusercontent.com/argoproj/argo-cd/v3.4.5/manifests/core-install.yaml"
)
# The core install ships no API server, and it is the API server that
# normally creates the `default` AppProject - without this, the Application
# sits at InvalidSpecError("project default which does not exist") forever.
APPPROJECT = """
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: default
  namespace: argocd
spec:
  sourceRepos: ["*"]
  destinations:
    - server: "*"
      namespace: "*"
  clusterResourceWhitelist:
    - group: "*"
      kind: "*"
"""

APPLICATION = """
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mbt-deploy
  namespace: argocd
spec:
  project: default
  source:
    repoURL: http://gitea:3000/mbt-showcase/deploy.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
"""


class K3d:
    def __init__(self, name: str, network: str, workdir, host_aliases: dict) -> None:
        self.name = name
        self.network = network
        self.workdir = workdir
        self.host_aliases = host_aliases
        self.kubeconfig = None

    def sh(self, *cmd: str, timeout: int = 600, check: bool = True, input: str | None = None):
        env = os.environ.copy()
        if self.kubeconfig:
            env["KUBECONFIG"] = self.kubeconfig
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, input=input, check=False
        )
        if check:
            assert proc.returncode == 0, f"{cmd} failed:\n{proc.stdout}\n{proc.stderr}"
        return proc

    def kubectl(self, *args: str, **kwargs):
        return self.sh("kubectl", *args, **kwargs)

    def up(self) -> None:
        registries = self.workdir / "registries.yaml"
        registries.write_text('mirrors:\n  "zot:5000":\n    endpoint:\n      - http://zot:5000\n')
        # --host-alias feeds CoreDNS's NodeHosts: PODS cannot use docker's
        # embedded DNS (127.0.0.11 is loopback-scoped to the node), so the
        # compose services ArgoCD and the CronJobs reference are aliased by
        # their session IPs. Node-level resolution (containerd image pulls
        # via registries.yaml) works either way.
        alias_flags = [
            flag
            for service, ip in sorted(self.host_aliases.items())
            for flag in ("--host-alias", f"{ip}:{service}")
        ]
        self.sh(
            "k3d",
            "cluster",
            "create",
            self.name,
            "--network",
            self.network,
            "--servers",
            "1",
            "--no-lb",
            *alias_flags,
            "--registry-config",
            str(registries),
            "--kubeconfig-update-default=false",
            "--kubeconfig-switch-context=false",
            "--wait",
            timeout=900,
        )
        kubeconfig = self.sh("k3d", "kubeconfig", "write", self.name).stdout.strip()
        self.kubeconfig = kubeconfig
        # Copy the ArgoCD images from the host daemon into the cluster's
        # containerd when present - best-effort (a cold host pulls from quay
        # inside the cluster instead, just slower).
        self.sh(
            "k3d",
            "image",
            "import",
            "-c",
            self.name,
            "quay.io/argoproj/argocd:v3.4.5",
            "public.ecr.aws/docker/library/redis:8.2.3-alpine",
            check=False,
            timeout=600,
        )

    def down(self) -> None:
        subprocess.run(
            ["k3d", "cluster", "delete", self.name],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    def wait_for(self, *args: str, timeout_s: int = 300, expect: str | None = None) -> str:
        import time

        end = time.time() + timeout_s
        last = ""
        while time.time() < end:
            proc = self.kubectl(*args, check=False)
            last = proc.stdout.strip()
            if proc.returncode == 0 and (expect is None or expect in last):
                return last
            time.sleep(5)
        pytest.fail(f"kubectl {args} never satisfied ({expect=}); last:\n{last}")


@pytest.fixture(scope="module")
def cd(showcase_ci, tmp_path_factory: pytest.TempPathFactory):
    for binary in ("k3d", "kubectl"):
        if shutil.which(binary) is None:
            pytest.fail(f"MBT_LIVE_SHOWCASE_K3D=1 but {binary} is not on PATH")

    ci = showcase_ci
    ci.ensure_seeded()

    def container_ip(service: str) -> str:
        proc = subprocess.run(
            [
                "docker",
                "inspect",
                f"{ci.stack.project_name}-{service}-1",
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0 and proc.stdout.strip(), (service, proc.stderr)
        return proc.stdout.strip()

    cluster = K3d(
        name=ci.stack.project_name.replace("mbt-show-", "mbt-k3d-"),
        network=ci.stack.env["SHOWCASE_NETWORK"],
        workdir=tmp_path_factory.mktemp("showcase-k3d"),
        host_aliases={s: container_ip(s) for s in ("gitea", "zot", "seaweedfs", "mlflow")},
    )
    try:
        cluster.up()
        cluster.kubectl("create", "namespace", "argocd")
        # --server-side: ArgoCD v3's applicationsets CRD blows past the
        # 262144-byte last-applied-configuration annotation limit that
        # client-side apply would write.
        cluster.kubectl("apply", "--server-side", "-n", "argocd", "-f", ARGOCD_CORE, timeout=600)
        cluster.wait_for(
            "get",
            "-n",
            "argocd",
            "deployment/argocd-repo-server",
            "-o",
            "jsonpath={.status.readyReplicas}",
            expect="1",
            timeout_s=600,
        )
        cluster.kubectl("apply", "-f", "-", input=APPPROJECT)
        cluster.kubectl("apply", "-f", "-", input=APPLICATION)
        yield {"ci": ci, "k3d": cluster}
    finally:
        # Same post-mortem escape hatch as the compose stack.
        if os.environ.get("MBT_SHOWCASE_KEEP") == "1":
            print(f"\nMBT_SHOWCASE_KEEP=1: k3d cluster {cluster.name} left running")
        else:
            cluster.down()


def _refresh(cluster: K3d) -> None:
    cluster.kubectl(
        "-n",
        "argocd",
        "annotate",
        "application/mbt-deploy",
        "argocd.argoproj.io/refresh=normal",
        "--overwrite",
    )


def _cronjob_image(cluster: K3d) -> str:
    return cluster.kubectl(
        "get",
        "-n",
        "mbt",
        "cronjob/mbt-score",
        "-o",
        "jsonpath={.spec.jobTemplate.spec.template.spec.containers[0].image}",
    ).stdout.strip()


def test_argocd_syncs_deploy_repo_and_pull_works(cd) -> None:
    ci, cluster = cd["ci"], cd["k3d"]
    _refresh(cluster)
    cluster.wait_for("get", "-n", "mbt", "cronjob/mbt-score", timeout_s=420)

    # The synced CronJob is pinned to the SAME digest CI pinned for Airflow,
    # spelled with the in-network registry host.
    digest = ci.images_env()["IMAGE"].split("@", 1)[1]
    image = _cronjob_image(cluster)
    assert image == f"zot:5000/mbt/churn@{digest}", (image, digest)

    # Insecure-HTTP pull from zot inside the cluster: run the unit.
    cluster.kubectl(
        "run",
        "pull-probe",
        "-n",
        "mbt",
        f"--image={image}",
        "--restart=Never",
        "--command",
        "--",
        "mbt",
        "--version",
    )
    cluster.wait_for(
        "get",
        "-n",
        "mbt",
        "pod/pull-probe",
        "-o",
        "jsonpath={.status.phase}",
        expect="Succeeded",
        timeout_s=600,
    )


def test_digest_bump_rolls_the_cronjob(cd) -> None:
    ci, cluster = cd["ci"], cd["k3d"]

    # In a full-suite session the CI module baked twice (bootstrap + the
    # gate merge): roll BACK to the older digest - a real, pullable unit,
    # exactly what a `git revert` rollback looks like to CD. Standalone
    # sessions have one bake; the roll then targets a synthetic digest
    # (the suspended CronJob never pulls, so the CD mechanics still prove).
    current = _cronjob_image(cluster)
    bumps = [
        c
        for c in reversed(ci.deploy_commits())
        if c["commit"]["message"].startswith("deploy: unit digest from ")
    ]
    old_digest = ci.images_env(ref=bumps[0]["sha"])["IMAGE"].split("@", 1)[1]
    if old_digest == current.split("@", 1)[1]:
        old_digest = "sha256:" + "0" * 64

    clone = ci.fresh_clone("deploy-roll", repo="deploy")
    cronjob = clone / "k8s" / "score-cronjob.yaml"
    text = cronjob.read_text()
    rolled = text.replace(current.split("@", 1)[1], old_digest)
    assert rolled != text
    cronjob.write_text(rolled)
    run_git(clone, "add", "k8s/score-cronjob.yaml")
    run_git(clone, "commit", "-m", "roll back to the previous unit digest")
    run_git(clone, "push", "origin", "main")

    _refresh(cluster)
    import time

    end = time.time() + 420
    while time.time() < end:
        if _cronjob_image(cluster) == f"zot:5000/mbt/churn@{old_digest}":
            return
        _refresh(cluster)
        time.sleep(10)
    pytest.fail(f"CronJob never rolled to {old_digest}; still {_cronjob_image(cluster)}")


def test_selfheal_recreates_deleted_cronjob(cd) -> None:
    cluster = cd["k3d"]
    cluster.kubectl("delete", "-n", "mbt", "cronjob/mbt-score")
    _refresh(cluster)
    cluster.wait_for("get", "-n", "mbt", "cronjob/mbt-score", timeout_s=420)
