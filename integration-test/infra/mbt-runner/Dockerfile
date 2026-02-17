FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy MBT source packages
COPY mbt-core/ /opt/mbt/mbt-core/
COPY mbt-sklearn/ /opt/mbt/mbt-sklearn/
COPY mbt-mlflow/ /opt/mbt/mbt-mlflow/
COPY mbt-postgres/ /opt/mbt/mbt-postgres/
COPY mbt-s3/ /opt/mbt/mbt-s3/

RUN pip install --no-cache-dir \
    /opt/mbt/mbt-core \
    /opt/mbt/mbt-sklearn \
    /opt/mbt/mbt-mlflow \
    /opt/mbt/mbt-postgres \
    /opt/mbt/mbt-s3

WORKDIR /opt/mbt/project

ENTRYPOINT ["mbt"]
