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


def test_the_extras_closure_agrees_with_uv_lock_on_shared_packages() -> None:
    """image-extras.txt is resolved AGAINST uv.lock's pins, so the two must
    agree on every package they both name - pip is handed both as constraint
    files and two different pins for one package is an unsatisfiable build.

    This is the drift alarm: a `uv lock` that moves numpy/pandas/scikit-learn
    invalidates the committed closure, and it should fail here - in the fast
    suite, in a second, naming the fix - rather than five minutes into a docker
    build in an opt-in tier.
    """
    extras = _requirements(EXTRAS_TXT)
    assert extras, "image-extras.txt has no pins; run scripts/lock_image_extras.sh"

    import subprocess

    exported = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-emit-workspace",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if exported.returncode != 0:  # pragma: no cover - uv is always present in CI
        import pytest

        pytest.skip(f"uv export unavailable: {exported.stderr.strip()[:200]}")

    locked: dict[str, str] = {}
    for line in exported.stdout.splitlines():
        match = re.match(r"^([A-Za-z0-9._-]+)==([^;\s]+)", line.strip())
        if match:
            locked.setdefault(_canon(match.group(1)), match.group(2))

    disagreements = {
        name: (locked[name], extras[name])
        for name in locked.keys() & extras.keys()
        if locked[name] != extras[name]
    }
    assert not disagreements, (
        f"uv.lock and image-extras.txt disagree on {disagreements} (locked, extras). "
        f"pip gets both as constraint files, so the runner image cannot build. "
        f"Regenerate: examples/showcase/scripts/lock_image_extras.sh"
    )


def test_the_closure_pins_the_transitives_uv_lock_cannot_see() -> None:
    """The point of the file: packages nothing else in the repo pins.
    statsmodels is the one that actually broke a nightly."""
    extras = _requirements(EXTRAS_TXT)
    for package in ("evidently", "jupyterlab", "statsmodels", "plotly", "nltk"):
        assert package in extras, f"image-extras.txt does not pin {package}"


# -- S3 credential defaults (three files, one truth) ---------------------------

COMPOSE = REPO_ROOT / "examples" / "showcase" / "compose" / "docker-compose.yml"
S3_CONFIG = REPO_ROOT / "examples" / "showcase" / "compose" / "seaweedfs" / "s3_config.json"
MAKEFILE = REPO_ROOT / "examples" / "showcase" / "Makefile"
SHOWCASE_UTILS = REPO_ROOT / "tests" / "showcase_utils.py"


def test_host_run_s3_credentials_match_the_stack() -> None:
    """The Snowflake plane runs on the HOST, so it cannot inherit the container
    env that every other target gets from compose - the Makefile and the test
    harness each carry their own fallback copy of the SeaweedFS credentials.

    That is three restatements of one fact, and getting one wrong fails LATE
    and confusingly: the dataset builds, the model trains, and only the
    artifact upload dies with `InvalidAccessKeyId` - which reads like a
    warehouse or network problem rather than a typo. (It happened: the
    host-run path shipped with an invented `mbtshowcase` key.)

    seaweedfs/s3_config.json is the authority; everything else must agree.
    """
    import json

    identity = json.loads(S3_CONFIG.read_text())["identities"][0]["credentials"][0]
    key, secret = identity["accessKey"], identity["secretKey"]

    compose = COMPOSE.read_text()
    assert f"${{SHOWCASE_S3_KEY:-{key}}}" in compose, f"compose default is not {key}"
    assert f"${{SHOWCASE_S3_SECRET:-{secret}}}" in compose, f"compose default is not {secret}"

    makefile = MAKEFILE.read_text()
    assert f"$(or $(SHOWCASE_S3_KEY),{key})" in makefile, (
        f"the Makefile's host-run fallback must be {key} (see HOST_MBT)"
    )
    assert f"$(or $(SHOWCASE_S3_SECRET),{secret})" in makefile, (
        f"the Makefile's host-run fallback must be {secret} (see HOST_MBT)"
    )

    utils = SHOWCASE_UTILS.read_text()
    assert f'os.environ.get("SHOWCASE_S3_KEY", "{key}")' in utils, (
        f"showcase_utils.host_env must fall back to {key}"
    )
    assert f'os.environ.get("SHOWCASE_S3_SECRET", "{secret}")' in utils, (
        f"showcase_utils.host_env must fall back to {secret}"
    )
