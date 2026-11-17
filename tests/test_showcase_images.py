"""Every image the showcase runs is built from this repo, and none is pulled.

The stack used to pull sixteen upstream images - thirteen from Docker Hub -
and Docker Hub is exactly what fails behind a TLS-intercepting office proxy
(`x509: certificate signed by unknown authority` from auth.docker.io) or
under its anonymous rate limit. Now each one is built by
examples/showcase/scripts/build_images.sh from images/<name>/Dockerfile, and
everything that names an image names one of those, under localhost/ so a
lookup of its origin can only ever hit the daemon's own loopback.

The guards are static reads, so they run in the fast tier: a new service, a
pipeline step or a version bump that reaches for a registry again fails here,
not on the first machine that cannot reach that registry.
"""

import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_DIR = REPO_ROOT / "examples" / "showcase"
IMAGES_DIR = SHOWCASE_DIR / "images"
BUILD_IMAGES = SHOWCASE_DIR / "scripts" / "build_images.sh"
COMPOSE = SHOWCASE_DIR / "compose" / "docker-compose.yml"
PIPELINES = sorted((SHOWCASE_DIR / "project" / ".woodpecker").glob("*.yml"))
UNIT_DOCKERFILE = SHOWCASE_DIR / "project" / "deploy" / "Dockerfile"
K8S_MANIFESTS = sorted((SHOWCASE_DIR / "deploy" / "k8s").glob("*.yaml"))
SOCK_PROBE = SHOWCASE_DIR / "scripts" / "docker_sock_gid.sh"
DOCTOR = SHOWCASE_DIR / "scripts" / "doctor.py"
MAKEFILE = SHOWCASE_DIR / "Makefile"


def _built_tags() -> dict[str, str]:
    """{image dir name: the tag build_images.sh builds it as}, asked of the
    script itself so the test and the build cannot disagree on the rule."""
    names = sorted(path.parent.name for path in IMAGES_DIR.glob("*/Dockerfile"))
    proc = subprocess.run(
        ["bash", str(BUILD_IMAGES), "--tags", *names],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    tags = proc.stdout.split()
    assert len(tags) == len(names), proc.stdout
    return dict(zip(names, tags, strict=True))


def _from_lines(dockerfile: Path) -> list[tuple[str, str | None]]:
    """(image reference, stage alias) for every FROM in a Dockerfile."""
    found = []
    for line in dockerfile.read_text().splitlines():
        match = re.match(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line, re.I)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def _pipeline_images(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text())
    images = [step["image"] for step in (config.get("clone") or {}).values()]
    steps = config.get("steps") or []
    if isinstance(steps, dict):
        steps = list(steps.values())
    images += [step["image"] for step in steps]
    return images


def _scalar(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(), re.M)
    assert match, f"{path.relative_to(REPO_ROOT)} no longer matches {pattern!r}"
    return match.group(1)


def _image_references() -> list[tuple[str, str]]:
    """(where, image) for every place the showcase names an image to run or
    build FROM, Dockerfile stages excluded."""
    refs = []
    for name, service in yaml.safe_load(COMPOSE.read_text())["services"].items():
        refs.append((f"compose service {name}", service["image"]))
    for path in PIPELINES:
        refs += [(f"pipeline {path.name}", image) for image in _pipeline_images(path)]
    for dockerfile in [*sorted(IMAGES_DIR.glob("*/Dockerfile")), UNIT_DOCKERFILE]:
        stages: set[str] = set()
        for image, alias in _from_lines(dockerfile):
            resolved = {image.replace("${TARGETARCH}", arch) for arch in ("amd64", "arm64")}
            if image != "scratch" and not resolved <= stages:
                refs.append((f"FROM in {dockerfile.relative_to(SHOWCASE_DIR)}", image))
            if alias:
                stages.add(alias)
    refs.append(("docker_sock_gid.sh PROBE_IMAGE", _scalar(SOCK_PROBE, r'^PROBE_IMAGE="([^"]+)"$')))
    refs.append(("doctor.py PROBE_IMAGE", _scalar(DOCTOR, r'^PROBE_IMAGE = "([^"]+)"$')))
    refs.append(("doctor.py RUNNER_IMAGE", _scalar(DOCTOR, r'^RUNNER_IMAGE = "([^"]+)"$')))
    refs.append(("Makefile RUNNER_IMAGE", _scalar(MAKEFILE, r"^RUNNER_IMAGE := (\S+)$")))
    return refs


def test_every_image_the_showcase_runs_is_one_this_repo_builds() -> None:
    """Compose services, Woodpecker steps and clone plugin, every Dockerfile's
    FROM (the deployable unit's included) and the probes all name an image
    build_images.sh builds, at exactly the tag it builds - so `make image`
    leaves nothing for docker to fetch from a registry, and a version bump in
    a Dockerfile that misses a consumer fails here instead of as a pull."""
    built = set(_built_tags().values())
    wrong = [(where, image) for where, image in _image_references() if image not in built]
    assert not wrong, (
        f"these name an image build_images.sh does not build (it builds {sorted(built)}): {wrong}"
    )


def test_the_deployable_unit_is_pulled_only_from_the_stacks_own_registry() -> None:
    """The k8s CronJob runs the unit prod-build baked from this repo and
    pushed to the stack's zot; that is the one image anything pulls."""
    for path in K8S_MANIFESTS:
        for image in re.findall(r"^\s*image:\s*(\S+)", path.read_text(), re.M):
            assert image.startswith("zot:5000/mbt/churn"), (path.name, image)


def test_every_dockerfile_builds_from_scratch_or_a_showcase_image() -> None:
    """The root image unpacks a rootfs onto scratch; every other image starts
    from a showcase image, so no FROM line can name a registry image."""
    tags = _built_tags()
    for name in tags:
        dockerfile = IMAGES_DIR / name / "Dockerfile"
        stages: set[str] = set()
        for image, alias in _from_lines(dockerfile):
            resolved = {image.replace("${TARGETARCH}", arch) for arch in ("amd64", "arm64")}
            assert image == "scratch" or resolved <= stages or image in tags.values(), (
                f"images/{name}/Dockerfile builds FROM {image}"
            )
            if alias:
                stages.add(alias)
    roots = [
        name for name in tags if _from_lines(IMAGES_DIR / name / "Dockerfile")[-1][0] == "scratch"
    ]
    assert roots == ["ubuntu"], f"final stage FROM scratch: {roots}"


def test_every_download_in_a_dockerfile_is_pinned_by_checksum() -> None:
    """Remote inputs come through `ADD --checksum=sha256:...` (BuildKit
    verifies them), never a bare curl/wget inside RUN, so an upstream asset
    replaced under the same URL fails the build instead of landing in it."""
    for dockerfile in sorted(IMAGES_DIR.glob("*/Dockerfile")):
        text = re.sub(r"\\\n", " ", dockerfile.read_text())
        for line in text.splitlines():
            instruction = line.strip()
            if re.match(r"^ADD\s", instruction, re.I) and re.search(r"https?://", instruction):
                assert re.search(r"--checksum=sha256:[0-9a-f]{64}\s", instruction), (
                    f"{dockerfile.relative_to(SHOWCASE_DIR)}: unpinned {instruction[:120]}"
                )
            if re.match(r"^RUN\s", instruction, re.I):
                assert not re.search(r"\b(curl|wget)\s+(-|https?://)", instruction), (
                    f"{dockerfile.relative_to(SHOWCASE_DIR)} downloads in RUN: {instruction[:120]}"
                )


def test_the_ubuntu_rootfs_is_pinned_for_both_architectures() -> None:
    """build_images.sh fetches the root image's rootfs outside docker and
    refuses it unless its sha256 matches; both architectures the showcase
    builds on (arm64 Macs, amd64 CI) need a pin."""
    text = (IMAGES_DIR / "ubuntu" / "Dockerfile").read_text()
    for arch in ("AMD64", "ARM64"):
        assert re.search(rf"^ARG ROOTFS_SHA256_{arch}=[0-9a-f]{{64}}$", text, re.M), arch


def test_compose_never_pulls() -> None:
    """`pull_policy: never` makes a missing image fail as "No such image" in
    a second, where compose's default would first ask a registry for it."""
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    pulling = sorted(
        name for name, service in services.items() if service.get("pull_policy") != "never"
    )
    assert not pulling, f"compose services without pull_policy: never: {pulling}"


def test_docker_run_never_pulls() -> None:
    """Every container the showcase's own scripts start runs a local image and
    says so, the docker-run counterpart of compose's pull_policy."""
    shells = [MAKEFILE, *sorted((SHOWCASE_DIR / "scripts").glob("*.sh"))]
    for path in shells:
        for line in path.read_text().splitlines():
            if re.search(r"\bdocker run\b", line) and not line.lstrip().startswith("#"):
                assert "--pull never" in line, f"{path.name}: {line.strip()}"
    calls = re.findall(r'docker\(\s*"run",[^)]*\)', DOCTOR.read_text(), re.S)
    assert calls, "doctor.py no longer starts a container; update this test"
    for call in calls:
        assert re.search(r'"--pull",\s*"never"', call), call


def test_the_probe_image_is_the_root_image() -> None:
    """docker_sock_gid.sh and doctor.py stat and df from inside a container,
    and both build their probe when it is missing. The root image is the only
    one cheap enough for that (a rootfs unpack, no registry, no apt) and is
    one every other build needs anyway."""
    root = _built_tags()["ubuntu"]
    assert _scalar(SOCK_PROBE, r'^PROBE_IMAGE="([^"]+)"$') == root
    assert _scalar(DOCTOR, r'^PROBE_IMAGE = "([^"]+)"$') == root


def test_woodpecker_trusts_the_clone_plugin_the_pipelines_use() -> None:
    """Woodpecker trusts its default clone plugin and flags any other clone
    image ("does not match allow list, netrc is not injected"); its built-in
    default is upstream's docker.io image. Compose points the default at the
    local build, and every pipeline must clone with exactly that image."""
    server = yaml.safe_load(COMPOSE.read_text())["services"]["woodpecker-server"]
    default = server["environment"]["WOODPECKER_DEFAULT_CLONE_PLUGIN"]
    for path in PIPELINES:
        clone = yaml.safe_load(path.read_text())["clone"]
        assert [step["image"] for step in clone.values()] == [default], path.name


#: Consumers of image names outside the showcase dir.
NAMES_ELSEWHERE = [
    REPO_ROOT / "tests" / "showcase_utils.py",
    *sorted((REPO_ROOT / ".github").rglob("*.yml")),
]


def test_no_reference_names_a_showcase_image_without_localhost() -> None:
    """A bare `mbt-showcase-runner:dev` means docker.io/library/mbt-showcase-
    runner, and the daemon goes there whenever it reaches for the image's
    origin: Woodpecker pulls a step image it cannot find, and a bake whose base
    has lost its layer blobs fetches them. Both then failed against Docker Hub
    (`insufficient_scope`, or x509 behind a TLS-intercepting proxy) instead of
    failing locally. Markdown is not scanned: prose quotes the old names."""
    names = "|".join(sorted(_built_tags(), key=len, reverse=True))
    pattern = re.compile(rf"mbt-showcase-(?:{names})(?![\w-])(?::[\w.-]+)?")
    sources = [
        path
        for path in sorted(SHOWCASE_DIR.rglob("*"))
        if path.is_file()
        and path.suffix not in {".md", ".pyc"}
        and path.name != ".env"
        and "__pycache__" not in path.parts
    ]
    for path in [*sources, *NAMES_ELSEWHERE]:
        text = path.read_text(errors="ignore")
        for match in pattern.finditer(text):
            if text[: match.start()].endswith("localhost/"):
                continue
            line = text.count("\n", 0, match.start()) + 1
            raise AssertionError(
                f"{path.relative_to(REPO_ROOT)}:{line} names {match.group(0)} without "
                f"localhost/, which resolves to docker.io/library/{match.group(0)}"
            )
