# Redis for ArgoCD in the k3d + ArgoCD tier, in place of the docker official
# redis image ArgoCD's manifest names (mirrored on public.ecr.aws): Ubuntu's
# redis-server from the base image's archive snapshot. ArgoCD runs it with
# arguments only (`--save "" --appendonly no --requirepass ...`), so the
# server binary is the entrypoint, as upstream's entrypoint makes it. Its pod
# runs as uid 999, the official image's redis user, which owns /data, the
# working directory redis looks for a dump in at start; the redis user is
# created with that uid before the package would pick another.
ARG VERSION=8.0.5

FROM localhost/mbt-showcase-base:26.04.1
ARG VERSION
RUN groupadd --system --gid 999 redis \
    && useradd --system --uid 999 --gid redis --home-dir /var/lib/redis --shell /usr/sbin/nologin redis \
    && apt-get install -y --no-install-recommends "redis-server=5:${VERSION}-*" \
    && apt-get clean \
    && mkdir /data \
    && chown redis:redis /data \
    && redis-server --version
USER 999
WORKDIR /data
EXPOSE 6379
ENTRYPOINT ["redis-server"]
