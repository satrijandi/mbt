# The docker CLI for prod-build's bake step, which drives the host daemon
# through the mounted socket: the static release binary from
# download.docker.com plus the buildx plugin, the two pieces of the docker
# official `docker:<version>-cli` image the step uses (`docker build` goes
# through buildx, as it does there). download.docker.com publishes no
# checksums; these are sha256 of the archives as fetched.
ARG VERSION=29.8.0
ARG BUILDX_VERSION=0.37.0
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION BUILDX_VERSION
ADD --checksum=sha256:cc21815cf1e2efed867dc9c8b96b46ffed8ea176ffab32b0aacb54726ded8f25 \
    https://download.docker.com/linux/static/stable/x86_64/docker-${VERSION}.tgz /docker.tgz
ADD --checksum=sha256:ae43fa08c796b44efc86d7a63c55f73f7c35f3101188dea7bf93bcd6f99577ba \
    https://github.com/docker/buildx/releases/download/v${BUILDX_VERSION}/buildx-v${BUILDX_VERSION}.linux-amd64 /docker-buildx

FROM scratch AS fetch-arm64
ARG VERSION BUILDX_VERSION
ADD --checksum=sha256:1462a696be6029bd478d7d60d7f3c31cdd15affd1178a4a278aaf4a1d1b7f8b5 \
    https://download.docker.com/linux/static/stable/aarch64/docker-${VERSION}.tgz /docker.tgz
ADD --checksum=sha256:d263ce31bd2c9e9210aaa2c7537c67802bccabcd342e4c9fe4907085ddb41aa5 \
    https://github.com/docker/buildx/releases/download/v${BUILDX_VERSION}/buildx-v${BUILDX_VERSION}.linux-arm64 /docker-buildx

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN --mount=from=fetch,target=/fetch \
    set -eux; \
    tar -xzf /fetch/docker.tgz -C /usr/local/bin --strip-components=1 docker/docker; \
    mkdir -p /usr/local/libexec/docker/cli-plugins; \
    install -m 0755 /fetch/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx; \
    docker --version; \
    docker buildx version
CMD ["bash"]
