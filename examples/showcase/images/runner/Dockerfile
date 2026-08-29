# The showcase's single "runner" image: Jupyter kernel, Spark master/worker
# runtime, MLflow server, and every mbt invocation all run THIS environment,
# which makes ADR-19 env_digest verification hold by construction.
#
# Built by examples/showcase/scripts/build_image.sh, which stages the workspace
# wheels (uv build --all-packages) and a constraints file derived from uv.lock
# with the sparkling fork's pins (pyspark 3.5.x + h2o matched to the
# h2o-pysparkling embedded backend) into the build context.
# Digest-pinned (not a floating tag) so the runner base is reproducible, the
# way DESIGN.md's runner row claims; the tag stays for humans. Re-pin with
#   docker buildx imagetools inspect python:3.11-slim-bookworm
FROM python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

# JRE 17 satisfies Spark 3.5 and H2O; procps because Spark scripts call ps;
# curl for healthchecks, the s3a jar fetch below, and CI alert webhooks; git
# for the CI state-branch scripts (fetch_state/publish_state) and the DS
# clone loop; libgomp1 is LightGBM's OpenMP runtime (absent from -slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless curl procps git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY dist/ /opt/mbt/dist/
COPY constraints.txt /opt/mbt/constraints.txt
COPY image-extras.txt /opt/mbt/image-extras.txt

# The sparkling extra resolves the whole env to pyspark 3.5.x (ADR-17);
# constraints.txt pins everything else to uv.lock versions. The cache mount
# survives rebuilds so the ~800MB of JVM sdists download once; timeouts are
# generous because pypi serves those sdists slowly.
#
# mbt-lightgbm powers the wide cadence's feature-selection probe and the
# ds-helper funnel in scripts/select_features.py; evidently drives the
# feature-stability gates in scripts/evidently_gate.py and jupyterlab the DS
# inner loop. Those last two are showcase-image deps, never mbt deps, so
# uv.lock says nothing about them or their closure - image-extras.txt pins it
# (source of truth: image-extras.in; regenerate with scripts/lock_image_extras.sh).
# Without it this layer resolved fresh against PyPI on every build and
# inherited whatever upstream had published that morning, which is exactly how
# a sdist-only statsmodels release took the nightly down against an image that
# deliberately ships no compiler.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 300 --retries 10 \
    --find-links /opt/mbt/dist \
    --constraint /opt/mbt/constraints.txt \
    --constraint /opt/mbt/image-extras.txt \
    "mbt-core[s3]" "mbt-h2o[sparkling]" mbt-spark mbt-xgboost mbt-lightgbm \
    mbt-mlflow evidently \
    jupyterlab

# s3a support for the SeaweedFS lake: hadoop-aws must match the hadoop client
# libs bundled with pyspark 3.5.x (3.3.4), and the AWS SDK bundle must match
# what hadoop-aws 3.3.4 was built against.
RUN SPARK_PKG_DIR="$(python -c 'import pathlib, pyspark; print(pathlib.Path(pyspark.__file__).parent)')" \
    && ln -s "$SPARK_PKG_DIR" /opt/spark \
    && curl -fsSL -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
        https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    && curl -fsSL -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar \
        https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
# Spark daemons must log to the console (docker), not $SPARK_HOME/logs.
ENV SPARK_NO_DAEMONIZE=1

COPY entrypoint.sh /usr/local/bin/showcase-entrypoint
RUN chmod +x /usr/local/bin/showcase-entrypoint

WORKDIR /workspace
ENTRYPOINT ["showcase-entrypoint"]
CMD ["bash"]
