# The showcase's single "runner" image: Jupyter kernel, Spark master/worker
# runtime, MLflow server, and every mbt invocation all run THIS environment,
# which makes ADR-19 env_digest verification hold by construction.
#
# Built by examples/showcase/scripts/build_images.sh, which stages the
# workspace wheels (uv build --all-packages) and a constraints file derived
# from uv.lock with the sparkling fork's pins (pyspark 3.5.x + h2o matched to
# the h2o-pysparkling embedded backend) into the build context. Its base is
# the showcase's own CPython image, built from this repo like every other
# image the stack runs (images/python).
FROM localhost/mbt-showcase-python:3.11.17

# JRE 17 satisfies Spark 3.5 and H2O; procps because Spark scripts call ps;
# curl for healthchecks and CI alert webhooks; git for the CI state-branch
# scripts (fetch_state/publish_state) and the DS clone loop; libgomp1 is
# LightGBM's OpenMP runtime. apt resolves against the base image's pinned
# archive snapshot, whose package lists ship with it.
RUN apt-get install -y --no-install-recommends openjdk-17-jre-headless curl procps git libgomp1 \
    && apt-get clean

COPY dist/ /opt/mbt/dist/
COPY constraints.txt /opt/mbt/constraints.txt
COPY image-extras.txt /opt/mbt/image-extras.txt

# The sparkling extra resolves the whole env to pyspark 3.5.x (ADR-17);
# constraints.txt pins everything else to uv.lock versions. The cache mount
# survives rebuilds so the ~800MB of JVM sdists download once; timeouts are
# generous because pypi serves those sdists slowly.
#
# jupyterlab, for the DS inner loop, is a showcase-image dep and never an mbt
# dep, so uv.lock says nothing about it or its closure - image-extras.txt pins that
# (source of truth: image-extras.in; regenerate with scripts/lock_image_extras.sh).
# Without it this layer resolved fresh against PyPI on every build and
# inherited whatever upstream had published that morning, which is exactly how
# a sdist-only statsmodels release took the nightly down against an image that
# deliberately ships no compiler. setuptools is named so its pin there
# replaces the base CPython's bundled one, which has no pkg_resources for
# h2o-pysparkling to import (image-extras.in says why).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 \
    --find-links /opt/mbt/dist \
    --constraint /opt/mbt/constraints.txt \
    --constraint /opt/mbt/image-extras.txt \
    "mbt-core[s3]" "mbt-h2o[sparkling]" mbt-spark mbt-xgboost mbt-mlflow \
    jupyterlab setuptools

# pysparkling runs its version checks at import, so a missing dependency of its
# own (pkg_resources, above) fails here rather than inside the first cluster
# build.
RUN python -c "import pysparkling"

# s3a support for the SeaweedFS lake: hadoop-aws must match the hadoop client
# libs bundled with pyspark 3.5.x (3.3.4), and the AWS SDK bundle must match
# what hadoop-aws 3.3.4 was built against. Checksums are sha256 of the jars
# whose Maven Central .sha1 they match. They land in pyspark's own jars dir,
# which /opt/spark (SPARK_HOME) links to.
ADD --chmod=0644 --checksum=sha256:53f9ae03c681a30a50aa17524bd9790ab596b28481858e54efd989a826ed3a4a \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    /usr/local/lib/python3.11/site-packages/pyspark/jars/
ADD --chmod=0644 --checksum=sha256:873fe7cf495126619997bec21c44de5d992544aea7e632fdc77adb1a0915bae5 \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
    /usr/local/lib/python3.11/site-packages/pyspark/jars/
RUN SPARK_PKG_DIR="$(python -c 'import pathlib, pyspark; print(pathlib.Path(pyspark.__file__).parent)')" \
    && test -f "$SPARK_PKG_DIR/jars/hadoop-aws-3.3.4.jar" \
    && ln -s "$SPARK_PKG_DIR" /opt/spark

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
# Spark daemons must log to the console (docker), not $SPARK_HOME/logs.
ENV SPARK_NO_DAEMONIZE=1

COPY entrypoint.sh /usr/local/bin/showcase-entrypoint
RUN chmod +x /usr/local/bin/showcase-entrypoint

WORKDIR /workspace
ENTRYPOINT ["showcase-entrypoint"]
CMD ["bash"]
