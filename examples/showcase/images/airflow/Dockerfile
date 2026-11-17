# Airflow (the scheduler half of the CD loop): apache-airflow installed from
# PyPI under Airflow's own published constraints for this release, the way
# upstream builds its image, onto the showcase's CPython. Only what the
# showcase uses goes in: the Postgres metadata DB, the FAB auth manager
# compose selects, the standard provider (`@task`), and mbt-airflow with the
# docker SDK it drives the daemon with - baked here, where upstream's image
# installed the SDK at every container start from _PIP_ADDITIONAL_REQUIREMENTS.
#
# mbt-airflow draws the DAGs (ADR-33) and starts each task's container (ADR-34),
# built from this checkout's own wheels, as the runner image is. It brings
# mbt-core, which the DAG files import to read the manifest; the tasks
# themselves run mbt in the deployable unit, so nothing heavier is installed.
# Two constraint files apply: Airflow's pins its whole tree, and uv.lock's pins
# the rest of mbt's closure, minus whatever Airflow's already names, because
# Airflow's win there.
#
# Upstream's image contract otherwise: user airflow (50000, group 0),
# AIRFLOW_HOME=/opt/airflow, dumb-init as PID 1. Compose names each
# component's command (`airflow scheduler`, ...) and runs the one-shot init.
ARG VERSION=3.3.1

FROM localhost/mbt-showcase-python:3.11.17
ARG VERSION
ADD --checksum=sha256:9fe2d4a54b8ac8450f63ffce0b195de8d9027ddc21bc40653b62be91a63fa366 \
    https://raw.githubusercontent.com/apache/airflow/constraints-${VERSION}/constraints-3.11.txt \
    /opt/airflow-constraints.txt
RUN apt-get install -y --no-install-recommends curl dumb-init && apt-get clean
COPY dist/ /opt/mbt/dist/
COPY constraints.txt /opt/mbt/constraints-full.txt
RUN python - <<'PY'
import re
from pathlib import Path


def pinned(path):
    """(canonical name, line) for every requirement line of a constraints file."""
    for line in Path(path).read_text().splitlines():
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line.strip())
        if match:
            yield re.sub(r"[-_.]+", "-", match.group(1)).lower(), line


airflow = {name for name, _ in pinned("/opt/airflow-constraints.txt")}
# Airflow's constraints pin everything but Airflow's own distributions (core, task
# sdk, providers), and uv.lock's 3.4.0 would contradict the 3.3.1 this image installs.
rest = [
    line
    for name, line in pinned("/opt/mbt/constraints-full.txt")
    if name not in airflow and not name.startswith("apache-airflow")
]
Path("/opt/mbt/constraints.txt").write_text("\n".join(rest) + "\n")
PY
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 \
    --find-links /opt/mbt/dist \
    --constraint /opt/airflow-constraints.txt \
    --constraint /opt/mbt/constraints.txt \
    "apache-airflow[postgres]==${VERSION}" \
    apache-airflow-providers-fab \
    apache-airflow-providers-standard \
    "mbt-airflow[docker]"
RUN python -c "import mbt_airflow, docker"
RUN useradd --uid 50000 --gid 0 --home-dir /home/airflow --create-home --shell /bin/bash airflow \
    && mkdir -p /opt/airflow/logs /opt/airflow/dags \
    && chown -R airflow:0 /opt/airflow \
    && chmod -R g+w /opt/airflow

ENV AIRFLOW_HOME=/opt/airflow
USER 50000
WORKDIR /opt/airflow
EXPOSE 8080
ENTRYPOINT ["dumb-init", "--"]
CMD ["airflow", "version"]
