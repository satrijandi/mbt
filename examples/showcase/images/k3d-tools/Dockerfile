# k3d's helper container, which `k3d cluster create` always starts (and the
# k3d tier points K3D_IMAGE_TOOLS at), in place of ghcr.io/k3d-io/k3d-tools.
# k3d publishes it as an image only, so this compiles the tagged k3d source's
# tools/ module the way its Makefile does; the module vendors its
# dependencies, so the build is offline. Runtime as upstream's image: bash,
# getent (glibc), /app/k3d-tools as entrypoint.
ARG VERSION=5.9.0

FROM scratch AS source
ARG VERSION
ADD --checksum=sha256:969cce82c4871bb829be798655abe2b6709b77cc0f42f7ff69293c621dfbbab0 \
    https://github.com/k3d-io/k3d/archive/refs/tags/v${VERSION}.tar.gz /k3d.tar.gz

FROM localhost/mbt-showcase-base:26.04.1 AS build
ARG VERSION
RUN apt-get install -y --no-install-recommends golang-1.26-go
RUN --mount=from=source,target=/source \
    mkdir /src && tar -xzf /source/k3d.tar.gz -C /src --strip-components=1
WORKDIR /src/tools
RUN CGO_ENABLED=0 GOFLAGS=-mod=vendor GOPROXY=off GOTOOLCHAIN=local GOWORK=off \
    /usr/lib/go-1.26/bin/go build -trimpath \
    -ldflags "-w -s -X github.com/k3d-io/k3d/tools/version.Version=v${VERSION}" \
    -o /k3d-tools .

FROM localhost/mbt-showcase-base:26.04.1
COPY --from=build /k3d-tools /app/k3d-tools
WORKDIR /app
ENTRYPOINT ["/app/k3d-tools"]
