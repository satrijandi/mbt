FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
RUN usermod -aG docker airflow
USER airflow

RUN pip install --no-cache-dir apache-airflow-providers-docker>=3.10.0
