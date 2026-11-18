# Airflow (the scheduler half of the CD loop): upstream's apache/airflow image -
# its airflow user (50000, group 0), AIRFLOW_HOME=/opt/airflow, entrypoint and
# the `scheduler` / `dag-processor` / `api-server` commands compose names - with
# mbt-airflow and the docker SDK it drives the daemon with installed on top.
# Upstream's image installs that SDK at every container start from
# _PIP_ADDITIONAL_REQUIREMENTS; baking it here means a start needs no PyPI.
#
# mbt-airflow draws the DAGs (ADR-33) and starts each task's container (ADR-34),
# built from this checkout's own wheels, as the runner image is. It brings
# mbt-core, which the DAG files import to read the manifest; the tasks
# themselves run mbt in the deployable unit, so nothing heavier is installed.
# Two constraint files apply: Airflow's pins its whole tree, and uv.lock's pins
# the rest of mbt's closure, minus whatever Airflow's already names, because
# Airflow's win there. Airflow's is committed beside this file, so the build
# fetches nothing from GitHub, and it is Airflow's published file byte for byte:
#   source: https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.11.txt
#   sha256: 9fe2d4a54b8ac8450f63ffce0b195de8d9027ddc21bc40653b62be91a63fa366
# When VERSION moves, download the file for the new release the same way and
# update both lines; tests/test_showcase_image_pins.py fails until they agree.
#
# Built by examples/showcase/scripts/build_airflow_image.sh, which stages the
# wheels and uv.lock's pins into the build context (images/airflow/stage.sh) and
# tags the result localhost/mbt-showcase-airflow:<VERSION>.
ARG VERSION=3.3.1

FROM apache/airflow:${VERSION}-python3.11
ARG VERSION
USER root
COPY airflow-constraints.txt /opt/airflow-constraints.txt
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

USER airflow
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --find-links /opt/mbt/dist \
    --constraint /opt/airflow-constraints.txt \
    --constraint /opt/mbt/constraints.txt \
    "apache-airflow[postgres]==${VERSION}" \
    apache-airflow-providers-fab \
    apache-airflow-providers-standard \
    "mbt-airflow[docker]"
RUN python -c "import mbt_airflow, docker"
