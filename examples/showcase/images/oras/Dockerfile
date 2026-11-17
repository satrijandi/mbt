# oras, for prod-build's provenance step (manifest.json + run_results.json
# pushed to zot as an OCI artifact): the upstream release binary on the
# showcase base, with upstream's image contract - /bin/oras as entrypoint,
# workdir /workspace.
ARG VERSION=1.3.4
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
ADD --checksum=sha256:f27adb935022d94df8dc77719c322dda592c78a0d57a6f7dcdd8d900b248c454 \
    https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_amd64.tar.gz /oras.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
ADD --checksum=sha256:15702c6e3a4a56a8bd8ac5c17efdbcab56d9bada661ccbcf017f5b10c1d89399 \
    https://github.com/oras-project/oras/releases/download/v${VERSION}/oras_${VERSION}_linux_arm64.tar.gz /oras.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    tar -xzf /fetch/oras.tar.gz -C /bin oras \
    && oras version
WORKDIR /workspace
ENTRYPOINT ["/bin/oras"]
