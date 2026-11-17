# CPython for the images that run Python (the runner and Airflow): a
# python-build-standalone build - the relocatable CPython uv installs -
# unpacked into /usr/local, where the docker official python image puts its
# own build, so `python`, `python3` and `pip` resolve the same way. Ubuntu
# 26.04's own python3 is 3.14; the showcase stays on 3.11, the line the
# h2o-pysparkling 3.5 stack and the repo's JVM e2e tier run.
#
# Downloads are pinned by checksum (ADD --checksum), one stage per
# architecture; the daemon's own architecture picks the stage.
ARG VERSION=3.11.17
ARG PBS_RELEASE=20261001
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION PBS_RELEASE
ADD --checksum=sha256:7086a336e6ea0a49595cf891066ab6517156c85116f77fbc23b62c1d9e9b7d92 \
    https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${VERSION}%2B${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz \
    /python.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION PBS_RELEASE
ADD --checksum=sha256:26ee69bff5d6d4173360275fcf38d4d25228286b79a7d9b36ad39af970909e37 \
    https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${VERSION}%2B${PBS_RELEASE}-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz \
    /python.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
ARG VERSION
# Images install into this interpreter as root on purpose (there is no other
# user to install for), so pip's warning about it is noise.
ENV PIP_ROOT_USER_ACTION=ignore
RUN --mount=from=fetch,target=/fetch \
    set -eux; \
    tar -xzf /fetch/python.tar.gz -C /usr/local --strip-components=1; \
    ln -sf python3 /usr/local/bin/python; \
    ln -sf pip3 /usr/local/bin/pip; \
    test "$(python -c 'import platform; print(platform.python_version())')" = "$VERSION"; \
    python -m pip --version
