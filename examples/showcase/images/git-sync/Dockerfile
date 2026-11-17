# git-sync (hands the deploy repo's DAGs to Airflow): upstream publishes no
# binaries, only its own image, so this compiles the tagged source the way
# upstream's build/build.sh does - CGO off, version stamped, paths trimmed -
# with the Go toolchain line it builds with (Ubuntu's golang-1.25). The
# source tree vendors every module, so the build is offline: GOFLAGS forbids
# the module proxy and GOTOOLCHAIN forbids fetching another toolchain.
#
# Runtime as upstream's image: git, user git-sync (65533), HOME=/tmp,
# GITSYNC_ROOT=/git. Compose runs it as root to write the shared volume.
ARG VERSION=4.7.1

FROM scratch AS source
ARG VERSION
ADD --checksum=sha256:e575e37581869cd31d4752f50365fc0a94c3f9f95fb70c8b2f98262776077a3c \
    https://github.com/kubernetes/git-sync/archive/refs/tags/v${VERSION}.tar.gz /git-sync.tar.gz

FROM localhost/mbt-showcase-base:26.04.1 AS build
ARG VERSION
RUN apt-get install -y --no-install-recommends golang-1.25-go
RUN --mount=from=source,target=/source \
    mkdir /src && tar -xzf /source/git-sync.tar.gz -C /src --strip-components=1
WORKDIR /src
RUN CGO_ENABLED=0 GOFLAGS=-mod=vendor GOPROXY=off GOTOOLCHAIN=local \
    /usr/lib/go-1.25/bin/go build -trimpath \
    -ldflags "-s -w -X k8s.io/git-sync/pkg/version.VERSION=v${VERSION}" \
    -o /git-sync .

FROM localhost/mbt-showcase-base:26.04.1
RUN apt-get install -y --no-install-recommends git openssh-client \
    && apt-get clean \
    && groupadd --gid 65533 git-sync \
    && useradd --uid 65533 --gid git-sync --home-dir /tmp --no-create-home --shell /usr/sbin/nologin git-sync \
    && mkdir -m 02775 /git \
    && chown git-sync:git-sync /git
COPY --from=build /git-sync /git-sync
USER 65533:65533
ENV HOME=/tmp \
    GITSYNC_ROOT=/git
WORKDIR /tmp
ENTRYPOINT ["/git-sync"]
