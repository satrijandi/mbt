FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy MBT source packages
COPY mbt-core/ /opt/mbt/mbt-core/
COPY mbt-sklearn/ /opt/mbt/mbt-sklearn/
COPY mbt-mlflow/ /opt/mbt/mbt-mlflow/
COPY mbt-postgres/ /opt/mbt/mbt-postgres/
COPY mbt-s3/ /opt/mbt/mbt-s3/
COPY mbt-airflow/ /opt/mbt/mbt-airflow/

RUN chown -R airflow:root /opt/mbt

USER airflow

# Pin SQLAlchemy <2.0 to stay compatible with Airflow
RUN pip install --no-cache-dir \
    "sqlalchemy>=1.4,<2.0" \
    /opt/mbt/mbt-core \
    /opt/mbt/mbt-sklearn \
    /opt/mbt/mbt-mlflow \
    /opt/mbt/mbt-postgres \
    /opt/mbt/mbt-s3 \
    /opt/mbt/mbt-airflow
