#!/usr/bin/env bash
# Build every image the showcase runs, from this checkout, into the local
# docker daemon. The stack pulls no image from any registry: compose, the
# Woodpecker pipelines and the deployable unit's FROM name only
# localhost/mbt-showcase-* images, and each is built here from
# examples/showcase/images/<name>/Dockerfile. What a build fetches is pinned:
# release artifacts by sha256 (the Ubuntu rootfs below, ADD --checksum in the
# Dockerfiles), apt packages by one dated archive snapshot (images/base), and
# PyPI packages by constraints files.
#
# The localhost/ names keep it that way. A bare name means
# docker.io/library/<name>, and whenever the daemon went to an image's
# origin - a Woodpecker step whose image is missing, or a bake whose base has
# lost its layer blobs - it asked Docker Hub for an image that was never
# published, and failed with `insufficient_scope`, or with an x509 error
# behind a TLS-intercepting proxy. localhost/ sends that lookup to the
# daemon's own loopback, which Go's HTTP client never proxies, so it fails in
# a second, locally, whatever the network is doing.
#
# Tags: localhost/mbt-showcase-<name>:<VERSION>, VERSION being the
# `ARG VERSION=` default at the top of that Dockerfile - the upstream version
# the image packages. The runner is the exception, localhost/mbt-showcase-
# runner:dev: it is built from this checkout's own wheels.
#
# Each image is built after the localhost/mbt-showcase-* images its FROM lines
# name. Docker's build cache makes an unchanged image a few seconds' no-op,
# except the images built from this checkout's own wheels (the runner, and
# Airflow, which imports mbt-airflow), whose builds stage fresh wheels that
# would bust the cache every time. Those are skipped instead when the existing
# image's content label matches a hash of everything that shapes it (package
# sources and pyprojects, uv.lock, the image's own directory, and the
# Dockerfiles it builds FROM). A bare tag-existence check once let week-old
# wheels pass as "the current build", and every in-container mbt assertion
# silently tested old code.
# images/<name>/stage.sh stages such an image's build context, in a cache dir
# OUTSIDE the repo (the repo-root pytest session guard forbids new repo-root
# entries, and dist/ churn in-tree would be noise anyway).
#
# With no NAME, every image the stack and its pipelines run is built. The
# k3d + ArgoCD tier's images (ON_DEMAND) are not: only that opt-in tier runs
# them, and tests/test_showcase_k3d.py builds them by name.
#
# Usage: build_images.sh [--force] [NAME...]  NAMEs (default: the stack's
#                                             images), plus what they build
#                                             FROM; --force rebuilds the
#                                             staged images regardless
#        build_images.sh --print-hash [NAME]  a staged image's content hash
#                                             (default: the runner's)
#        build_images.sh --tags [NAME...]     the tags NAMEs (default: the
#                                             stack's images) build as
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGES_DIR="$REPO_ROOT/examples/showcase/images"
CACHE_DIR="${MBT_SHOWCASE_CACHE:-$HOME/.cache/mbt-showcase}"
PREFIX="localhost/mbt-showcase-"
RUNNER_TAG="${PREFIX}runner:dev"
CONTENT_LABEL="mbt.showcase.content"
ON_DEMAND="k3s k3d-tools argocd redis"

die() {
    echo "build_images.sh: $*" >&2
    exit 1
}

tag_of() {
    local name="$1" version
    [ -f "$IMAGES_DIR/$name/Dockerfile" ] || die "no image named $name (no $IMAGES_DIR/$name/Dockerfile)"
    if [ "$name" = runner ]; then
        echo "$RUNNER_TAG"
        return
    fi
    version="$(sed -nE 's/^ARG VERSION=(.+)$/\1/p' "$IMAGES_DIR/$name/Dockerfile" | head -1)"
    [ -n "$version" ] || die "$name/Dockerfile declares no ARG VERSION="
    echo "$PREFIX$name:$version"
}

# The showcase images NAME's Dockerfile builds FROM.
deps_of() {
    sed -nE "s#^FROM ${PREFIX}([a-z0-9-]+):.*#\1#p" "$IMAGES_DIR/$1/Dockerfile" | sort -u
}

# Hash of everything that shapes a staged image (NAME is the runner or
# airflow). Package tests/READMEs stay out on purpose: they never reach the
# installed wheels, and hashing them would force pointless ~10-minute rebuilds.
content_hash() {
    python3 - "$REPO_ROOT" "$1" <<'EOF'
import hashlib
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
name = sys.argv[2]
images = root / "examples" / "showcase" / "images"
paths = [root / "uv.lock"]
paths += sorted((root / "packages").glob("*/pyproject.toml"))
for src in sorted((root / "packages").glob("*/src")):
    paths += sorted(
        p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
own = images / name
paths += sorted(
    p for p in own.rglob("*") if p.is_file() and "__pycache__" not in p.parts
)
# The images this one builds FROM, transitively.
pending = [own / "Dockerfile"]
while pending:
    text = pending.pop().read_text()
    for base in re.findall(r"^FROM localhost/mbt-showcase-([a-z0-9-]+):", text, re.M):
        parent = images / base / "Dockerfile"
        if parent not in paths:
            paths.append(parent)
            pending.append(parent)
digest = hashlib.sha256()
for path in paths:
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest()[:16])
EOF
}

sha256_of() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

daemon_arch() {
    local arch
    arch="$(docker version --format '{{.Server.Arch}}')" || die "the docker daemon is not reachable"
    case "$arch" in
        amd64 | arm64) echo "$arch" ;;
        *) die "no Ubuntu rootfs pinned for the daemon's architecture: $arch" ;;
    esac
}

# The Ubuntu rootfs tarball ADD unpacks, fetched and checked here because ADD
# unpacks only local archives (images/ubuntu/Dockerfile).
build_ubuntu() {
    local tag="$1" dockerfile="$IMAGES_DIR/ubuntu/Dockerfile" version arch upper want got ctx url
    version="${tag##*:}"
    arch="$(daemon_arch)"
    upper="$(echo "$arch" | tr '[:lower:]' '[:upper:]')"
    want="$(sed -nE "s/^ARG ROOTFS_SHA256_${upper}=(.+)$/\1/p" "$dockerfile")"
    [ -n "$want" ] || die "ubuntu/Dockerfile pins no ROOTFS_SHA256_${upper}"
    ctx="$CACHE_DIR/ubuntu-base-$version-$arch"
    if [ ! -f "$ctx/rootfs.tar.gz" ] || [ "$(sha256_of "$ctx/rootfs.tar.gz")" != "$want" ]; then
        url="https://cdimage.ubuntu.com/ubuntu-base/releases/$version/release/ubuntu-base-$version-base-$arch.tar.gz"
        echo "==> fetching $url"
        mkdir -p "$ctx"
        curl -fsSL --retry 5 -o "$ctx/rootfs.tar.gz.part" "$url"
        got="$(sha256_of "$ctx/rootfs.tar.gz.part")"
        if [ "$got" != "$want" ]; then
            rm -f "$ctx/rootfs.tar.gz.part"
            die "$url has sha256 $got; ubuntu/Dockerfile pins $want"
        fi
        mv "$ctx/rootfs.tar.gz.part" "$ctx/rootfs.tar.gz"
    fi
    docker build -t "$tag" -f "$dockerfile" "$ctx"
}

content_label() {
    docker image inspect --format '{{json .Config.Labels}}' "$1" 2>/dev/null \
        | python3 -c "import json,sys; print((json.load(sys.stdin) or {}).get('$CONTENT_LABEL',''))" 2>/dev/null \
        || true
}

build_staged() {
    local name="$1" tag="$2" hash existing staging
    hash="$(content_hash "$name")"
    existing="$(content_label "$tag")"
    if [ "$FORCE" != 1 ] && [ -n "$existing" ] && [ "$existing" = "$hash" ]; then
        echo "    up to date (content $hash; --force rebuilds anyway)"
        return
    fi
    # The runner keeps the cache dir it has always had; each staged image has its own.
    staging="$CACHE_DIR/build"
    [ "$name" = runner ] || staging="$CACHE_DIR/build-$name"
    "$IMAGES_DIR/$name/stage.sh" "$staging"

    echo "==> docker build $tag (content $hash)"
    docker build -t "$tag" --label "$CONTENT_LABEL=$hash" \
        --label "org.opencontainers.image.source=https://github.com/satrijandi/mbt" "$staging"
}

all_images() {
    local dockerfile
    for dockerfile in "$IMAGES_DIR"/*/Dockerfile; do
        basename "$(dirname "$dockerfile")"
    done
}

stack_images() {
    local name
    for name in $(all_images); do
        case " $ON_DEMAND " in *" $name "*) ;; *) echo "$name" ;; esac
    done
}

BUILT=" "

build() {
    local name="$1" dep tag
    case "$BUILT" in *" $name "*) return ;; esac
    for dep in $(deps_of "$name"); do
        build "$dep"
    done
    tag="$(tag_of "$name")"
    echo "==> $name: $tag"
    case "$name" in
        ubuntu) build_ubuntu "$tag" ;;
        runner | airflow) build_staged "$name" "$tag" ;;
        *) docker build -t "$tag" "$IMAGES_DIR/$name" ;;
    esac
    BUILT="$BUILT$name "
}

case "${1:-}" in
    --print-hash)
        content_hash "${2:-runner}"
        exit 0
        ;;
    --tags)
        shift
        [ "$#" -gt 0 ] || set -- $(stack_images)
        for name in "$@"; do
            tag_of "$name"
        done
        exit 0
        ;;
esac

FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
    shift
fi

[ "$#" -gt 0 ] || set -- $(stack_images)
for name in "$@"; do
    build "$name"
done
echo "==> built:${BUILT% }"
