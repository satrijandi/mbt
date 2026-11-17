# Woodpecker agent (CI): the upstream release binary on the showcase base,
# with upstream's image contract - root (it drives the mounted docker
# socket), the same ENV and the same `ping` healthcheck.
ARG VERSION=3.18.1
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:c20c24fc52bee0082ae49e197ba5d9ee8f005c5eb0cffe46d7c8957cacaea191 \
    https://github.com/woodpecker-ci/woodpecker/releases/download/v${VERSION}/woodpecker-agent_linux_amd64.tar.gz \
    /woodpecker.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:5e0a04ebc4669ca9d9672b96fb3f86f4ef2328aec2734c431ca0633da6de69ce \
    https://github.com/woodpecker-ci/woodpecker/releases/download/v${VERSION}/woodpecker-agent_linux_arm64.tar.gz \
    /woodpecker.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    tar -xzf /fetch/woodpecker.tar.gz -C /bin woodpecker-agent

ENV GODEBUG=netdns=go \
    WOODPECKER_IN_CONTAINER=true
EXPOSE 3000
HEALTHCHECK CMD ["/bin/woodpecker-agent", "ping"]
ENTRYPOINT ["/bin/woodpecker-agent"]
