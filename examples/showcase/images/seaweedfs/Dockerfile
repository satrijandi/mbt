# SeaweedFS (the S3 lake and the artifact store): the upstream release's
# `weed` binary on the showcase base. Upstream's image entrypoint adds
# `-logtostderr=true` (glog writes to files under /tmp otherwise, and
# `docker logs` stays empty) and the defaults compose's command already
# spells out (-dir=/data, the volume size and ceiling), so the ENTRYPOINT
# carries the first and compose the rest.
#
# Runs as root, the posture of every other showcase service that writes a
# volume; upstream's 4.x entrypoint chowns /data and drops to a `seaweed`
# user instead. curl serves compose's healthcheck.
ARG VERSION=4.47
ARG TARGETARCH

FROM scratch AS fetch-amd64
ARG VERSION
# Upstream publishes md5 sums only; this is the sha256 of the asset whose
# md5 (d327aa9fc73bfa7861fe3be4a4e9ebb7) matched.
ADD --checksum=sha256:31fb804858885f9e7f18b6d3b1da09e824baac3e6a55b5a62c3c4c77e6ed6d7d \
    https://github.com/seaweedfs/seaweedfs/releases/download/${VERSION}/linux_amd64.tar.gz /weed.tar.gz

FROM scratch AS fetch-arm64
ARG VERSION
# md5 a3e61f0c680039720fb7372ab7b0fd1a, as above.
ADD --checksum=sha256:ba5c9def9ff98f78becdf309a50a9f906922f6383a7c4c772d01ede3cc012e16 \
    https://github.com/seaweedfs/seaweedfs/releases/download/${VERSION}/linux_arm64.tar.gz /weed.tar.gz

FROM fetch-${TARGETARCH} AS fetch

FROM localhost/mbt-showcase-base:26.04.1
RUN apt-get install -y --no-install-recommends curl && apt-get clean
RUN --mount=from=fetch,target=/fetch \
    tar -xzf /fetch/weed.tar.gz -C /usr/local/bin weed \
    && weed version
WORKDIR /data
VOLUME /data
EXPOSE 8333 8888 9333
ENTRYPOINT ["/usr/local/bin/weed", "-logtostderr=true"]
