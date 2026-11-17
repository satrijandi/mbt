# Woodpecker server (CI): the upstream release binary on the showcase base,
# with upstream's image contract - user woodpecker (1000), state under
# /var/lib/woodpecker (compose mounts woodpecker-data there), the same ENV
# and the same `ping` healthcheck.
ARG VERSION=3.18.1
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:02798d4022dc2e291355f46e82dfe9d0fbd1a37244a8244f642b1d93c43edb35 \
    https://github.com/woodpecker-ci/woodpecker/releases/download/v${VERSION}/woodpecker-server_linux_amd64.tar.gz \
    /woodpecker.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:765e2f49f879a31d348e389f7188b02a875492f1d6e09fca28ddef9e65052a06 \
    https://github.com/woodpecker-ci/woodpecker/releases/download/v${VERSION}/woodpecker-server_linux_arm64.tar.gz \
    /woodpecker.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    tar -xzf /fetch/woodpecker.tar.gz -C /bin woodpecker-server \
    && groupadd --gid 1000 woodpecker \
    && useradd --uid 1000 --gid woodpecker --home-dir /home/woodpecker --shell /bin/sh woodpecker \
    && mkdir -p /var/lib/woodpecker \
    && chown woodpecker:woodpecker /var/lib/woodpecker

ENV GODEBUG=netdns=go \
    WOODPECKER_IN_CONTAINER=true \
    XDG_CACHE_HOME=/var/lib/woodpecker \
    XDG_DATA_HOME=/var/lib/woodpecker \
    SQLITE_TMPDIR=/var/lib/woodpecker
USER woodpecker
EXPOSE 8000 9000 80 443
HEALTHCHECK CMD ["/bin/woodpecker-server", "ping"]
ENTRYPOINT ["/bin/woodpecker-server"]
