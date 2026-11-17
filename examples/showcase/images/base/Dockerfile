# The userland every showcase service image starts from: the Ubuntu rootfs
# with apt pinned to ONE dated snapshot of the Ubuntu archive. An apt-get
# install below resolves the same package versions on any machine on any day,
# until SNAPSHOT is bumped on purpose - the apt counterpart of pinning an image
# by digest. The package lists stay in the image: a snapshot never changes, so
# they cannot go stale, and every image built on this one installs without an
# `apt-get update` of its own. Only the pockets and components the images use
# are listed (main and universe, release + updates + security), without
# translations, which keeps the lists near 100MB.
#
# snapshot.ubuntu.com serves https only and the rootfs ships no CA bundle, so
# ca-certificates comes first from the release pocket, which is frozen once a
# release ships (so that step is deterministic too). The upgrade then brings
# the rootfs's own packages up to the snapshot.
ARG VERSION=26.04.1
FROM localhost/mbt-showcase-ubuntu:26.04.1

ARG SNAPSHOT=20261016T000000Z
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

RUN set -eux; \
    sources=/etc/apt/sources.list.d/ubuntu.sources; \
    keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg; \
    echo 'Acquire::Languages "none";' > /etc/apt/apt.conf.d/99showcase; \
    printf '%s\n' 'Types: deb' 'URIs: http://archive.ubuntu.com/ubuntu/' 'Suites: resolute' \
        'Components: main' "Signed-By: $keyring" > "$sources"; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    printf '%s\n' 'Types: deb' "URIs: https://snapshot.ubuntu.com/ubuntu/${SNAPSHOT}/" \
        'Suites: resolute resolute-updates resolute-security' 'Components: main universe' \
        "Signed-By: $keyring" > "$sources"; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get clean
