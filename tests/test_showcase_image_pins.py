"""The showcase runner image's hardcoded pins must satisfy declared metadata.

`scripts/build_image.sh` pins the JVM trio by hand, because the sparkling fork
needs versions the dev lock deliberately does not carry (ADR-17). Those pins are
a second source of truth, and nothing checked it against the first: raising
mbt-h2o's `h2o` floor to 3.46.0.11 left the script pinning `h2o==3.46.0.6`, and
the two only met inside a docker build, as

    ERROR: Cannot install mbt-h2o[sparkling]==0.1.0 ...
    mbt-h2o[sparkling] 0.1.0 depends on h2o<3.46.0.12 and >=3.46.0.11
    The user requested (constraint) h2o==3.46.0.6

roughly five minutes into `make up`, on a machine with docker, in a tier that
is opt-in. test_showcase_infra.py does assert the h2o/pysparkling pairing, but
it is `live_showcase`-marked and needs the stack already built - it cannot see a
conflict that stops the image existing.

These are static reads of two files, so they run in the fast tier and fail in
the same second the pin and the specifier disagree.
"""

import re
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_IMAGE = REPO_ROOT / "examples" / "showcase" / "scripts" / "build_image.sh"
MBT_H2O = REPO_ROOT / "packages" / "mbt-h2o" / "pyproject.toml"


def _pin(name: str) -> tuple[str, str]:
    """(package, version) from a `NAME_PIN="pkg==1.2.3"` line in build_image.sh."""
    match = re.search(rf'^{name}="([A-Za-z0-9._-]+)==([^"]+)"', BUILD_IMAGE.read_text(), re.M)
    assert match is not None, f"{name} is no longer declared in {BUILD_IMAGE.name}"
    return match.group(1), match.group(2)


def _canon(name: str) -> str:
    """PEP 503 canonical form: `h2o-pysparkling-3.5` and `...-3-5` are one project."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(package: str, extra: str | None = None) -> SpecifierSet:
    project = tomllib.loads(MBT_H2O.read_text())["project"]
    requirements = (
        project["dependencies"] if extra is None else project["optional-dependencies"][extra]
    )
    for requirement in requirements:
        name = re.match(r"^([A-Za-z0-9._-]+)", requirement)
        assert name is not None
        if _canon(name.group(1)) == _canon(package):
            return SpecifierSet(requirement[name.end() :].strip())
    raise AssertionError(f"mbt-h2o does not declare {package}")


def test_the_pinned_h2o_satisfies_the_declared_range() -> None:
    """The regression that reached a user: a floor above the sparkling backend's
    embedded H2O makes `pip install mbt-h2o[sparkling]` ResolutionImpossible."""
    _, version = _pin("H2O_PIN")
    declared = _declared("h2o")
    assert declared.contains(Version(version)), (
        f"build_image.sh pins h2o=={version} but mbt-h2o declares h2o{declared}. "
        f"The showcase image cannot build. The floor is bounded above by the H2O "
        f"embedded in h2o-pysparkling-3.5, not by the advisory list."
    )


def test_the_pinned_pyspark_satisfies_the_sparkling_extra() -> None:
    _, version = _pin("PYSPARK_PIN")
    declared = _declared("pyspark", extra="sparkling")
    assert declared.contains(Version(version)), (
        f"build_image.sh pins pyspark=={version} but mbt-h2o[sparkling] declares pyspark{declared}"
    )


def test_the_pinned_pysparkling_satisfies_the_sparkling_extra() -> None:
    package, version = _pin("PYSPARKLING_PIN")
    declared = _declared(package, extra="sparkling")
    assert declared.contains(Version(version)), (
        f"build_image.sh pins {package}=={version} but mbt-h2o[sparkling] declares {declared}"
    )


def test_the_h2o_client_pin_matches_the_backend_pysparkling_embeds() -> None:
    """H2O requires the python client to equal the backend version exactly.
    test_showcase_infra.py proves this inside the built image; this catches a
    mismatched pair before anything is built."""
    _, h2o_version = _pin("H2O_PIN")
    _, pysparkling_version = _pin("PYSPARKLING_PIN")
    embedded = pysparkling_version.split(".post")[0]
    assert h2o_version == embedded, (
        f"h2o=={h2o_version} does not match the {embedded} backend embedded in "
        f"h2o-pysparkling-3-5=={pysparkling_version}; h2o.init() rejects a mismatch"
    )


# -- the non-mbt image deps and their pinned closure ---------------------------

RUNNER_DIR = REPO_ROOT / "examples" / "showcase" / "images" / "runner"
DOCKERFILE = RUNNER_DIR / "Dockerfile"
EXTRAS_IN = RUNNER_DIR / "image-extras.in"
EXTRAS_TXT = RUNNER_DIR / "image-extras.txt"


def _requirements(path: Path) -> dict[str, str]:
    """{canonical name: version} from a pinned requirements/constraints file.

    Markered lines (`numpy==2.4.6 ; python_full_version < '3.12'`) appear once
    per marker; the first wins, which is enough for an agreement check because
    both files are resolved for the same interpreter set.
    """
    pins: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        requirement = stripped.split(";", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9._-]+)==(.+)$", requirement)
        if match:
            pins.setdefault(_canon(match.group(1)), match.group(2).strip())
    return pins


def test_the_image_installs_only_extras_the_closure_pins() -> None:
    """Anything the Dockerfile pip-installs from PyPI that is not an mbt wheel
    must be declared in image-extras.in, or its closure is unpinned and the
    build inherits whatever upstream published that day - which is how a
    statsmodels release with macOS wheels only (Linux wheels followed four
    hours later) took down the 2026-08-27 nightly against an image that ships
    no compiler."""
    install = re.search(
        r"pip install (.*?)(?=\n\n|\nRUN|\nENV|\nCOPY)", DOCKERFILE.read_text(), re.S
    )
    assert install is not None, "could not find the pip install layer in the Dockerfile"
    tokens = install.group(1).replace("\\\n", " ").split()

    #: pip flags that consume the next token, so it is a value and not a package.
    takes_a_value = {"--timeout", "--retries", "--find-links", "--constraint", "-c"}
    requested: set[str] = set()
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            skip_next = token in takes_a_value
            continue
        # `"mbt-core[s3]"` -> mbt-core; `evidently==1.2` -> evidently
        name = token.strip('"').split("==")[0].split("[")[0]
        requested.add(_canon(name))
    from_pypi = {name for name in requested if not name.startswith("mbt-")}

    declared = {
        _canon(line.split("==")[0])
        for line in EXTRAS_IN.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert from_pypi == declared, (
        f"the Dockerfile installs {sorted(from_pypi)} from PyPI but image-extras.in "
        f"declares {sorted(declared)}; every non-mbt package must be declared there "
        f"so lock_image_extras.sh pins its closure"
    )


def _committed_lock_versions() -> dict[str, set[str]]:
    """{canonical name: every version uv.lock pins} from the COMMITTED lock.

    Deliberately the committed file rather than the working tree's, and read
    directly rather than through `uv export`, for one reason: the
    upstream-resolution tier runs `uv lock --upgrade` and then the fast suite,
    so in that tier the working tree's lock is a throwaway re-resolution that
    is never committed. Comparing the committed image closure against it fails
    by construction and says nothing - which is exactly what it did the first
    time this test met that tier.

    A set, not a version, because the lock legitimately pins several versions
    of one package: `[tool.uv] conflicts` forks numpy/scipy/pyspark/xgboost so
    mbt-h2o[sparkling] can hold Spark 3.5 (ADR-17). Any of them satisfies pip.
    """
    import subprocess
    import tomllib

    committed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "HEAD:uv.lock"],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = committed.stdout if committed.returncode == 0 else (REPO_ROOT / "uv.lock").read_text()

    versions: dict[str, set[str]] = {}
    for package in tomllib.loads(raw).get("package", []):
        versions.setdefault(_canon(package["name"]), set()).add(package["version"])
    return versions


def test_the_extras_closure_agrees_with_uv_lock_on_shared_packages() -> None:
    """image-extras.txt is resolved AGAINST uv.lock's pins, so the two must
    agree on every package they both name - pip is handed both as constraint
    files and two different pins for one package is an unsatisfiable build.

    This is the drift alarm: a commit that moves numpy/pandas/scikit-learn in
    the lock invalidates the committed closure, and it should fail here - in
    the fast suite, in a second, naming the fix - rather than five minutes into
    a docker build in an opt-in tier.
    """
    extras = _requirements(EXTRAS_TXT)
    assert extras, "image-extras.txt has no pins; run scripts/lock_image_extras.sh"

    locked = _committed_lock_versions()
    assert locked, "could not read any package pins out of uv.lock"

    disagreements = {
        name: (sorted(locked[name]), extras[name])
        for name in locked.keys() & extras.keys()
        if extras[name] not in locked[name]
    }
    assert not disagreements, (
        f"uv.lock and image-extras.txt disagree on {disagreements} (locked, extras). "
        f"pip gets both as constraint files, so the runner image cannot build. "
        f"Regenerate: examples/showcase/scripts/lock_image_extras.sh"
    )


def test_every_package_the_image_installs_is_pinned_somewhere() -> None:
    """The point of the two files: nothing the image installs resolves fresh.
    statsmodels is the one that actually broke a nightly; it arrives through
    evidently, which mbt-evidently made an mbt dependency, so uv.lock pins that
    closure now and image-extras.txt keeps only what mbt never depends on."""
    import tomllib

    extras = _requirements(EXTRAS_TXT)
    assert "jupyterlab" in extras, "image-extras.txt does not pin jupyterlab"
    locked = {
        _canon(package["name"])
        for package in tomllib.loads((REPO_ROOT / "uv.lock").read_text())["package"]
    }
    for package in ("evidently", "statsmodels", "plotly", "nltk"):
        assert package in locked, f"uv.lock does not pin {package}"
        assert package not in extras, f"{package} is pinned twice; drop it from image-extras.in"


# -- S3 credential defaults (two files, one truth) ----------------------------

COMPOSE = REPO_ROOT / "examples" / "showcase" / "compose" / "docker-compose.yml"
S3_CONFIG = REPO_ROOT / "examples" / "showcase" / "compose" / "seaweedfs" / "s3_config.json"


def test_compose_s3_credentials_match_the_seaweedfs_identity() -> None:
    """Every container gets its AWS_* from compose's ${SHOWCASE_S3_KEY:-...}
    defaults, and SeaweedFS accepts only the identity in s3_config.json. A
    mismatch fails LATE and confusingly: the dataset builds, the model trains,
    and only the artifact upload dies with `InvalidAccessKeyId`.

    seaweedfs/s3_config.json is the authority; compose must agree.
    """
    import json

    identity = json.loads(S3_CONFIG.read_text())["identities"][0]["credentials"][0]
    key, secret = identity["accessKey"], identity["secretKey"]

    compose = COMPOSE.read_text()
    assert f"${{SHOWCASE_S3_KEY:-{key}}}" in compose, f"compose default is not {key}"
    assert f"${{SHOWCASE_S3_SECRET:-{secret}}}" in compose, f"compose default is not {secret}"


# -- SeaweedFS capacity (must not follow the host's free disk) ----------------

SEED_LAKE = REPO_ROOT / "examples" / "showcase" / "bootstrap" / "churn_panel.py"
#: Volumes SeaweedFS grows for a no-replication collection on its first write.
SEAWEED_GROWTH_PER_COLLECTION = 7


def test_seaweedfs_capacity_does_not_follow_free_disk() -> None:
    """weed server's defaults make 1GB volumes and cap their number at free
    disk / volume size, and each bucket is a collection that grows 7 volumes
    on first write. On a docker disk with ~7GB free that cap was 7: seeding
    the lake took every slot, and every model upload to mbt-artifacts then
    failed with S3 InternalError ("No writable volumes and no free volumes
    left") - a whole `make demo` red over how full the laptop's disk was.

    The compose command pins both knobs instead, and the ceiling has to seat
    every collection the stack writes (each bucket plus the filer's own
    default collection) with a full growth step to spare.
    """
    import yaml

    command = yaml.safe_load(COMPOSE.read_text())["services"]["seaweedfs"]["command"]
    flags = dict(token.split("=", 1) for token in command.split() if "=" in token)

    assert "-master.volumeSizeLimitMB" in flags, "seaweedfs volume size is left to the default"
    assert int(flags["-master.volumeSizeLimitMB"]) <= 256, flags
    assert "-volume.max" in flags, "seaweedfs volume count is left to free disk"

    buckets = len(re.findall(r'^[A-Z_]+_BUCKET = "', SEED_LAKE.read_text(), re.M))
    assert buckets == 2, f"churn_panel.py creates {buckets} buckets; update this test"
    collections = buckets + 1  # plus the filer's default collection
    needed = (collections + 1) * SEAWEED_GROWTH_PER_COLLECTION
    assert int(flags["-volume.max"]) >= needed, (
        f"-volume.max={flags['-volume.max']} cannot seat {collections} collections "
        f"growing {SEAWEED_GROWTH_PER_COLLECTION} volumes each, plus one growth step"
    )
