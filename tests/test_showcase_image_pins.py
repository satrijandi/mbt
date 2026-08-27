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
