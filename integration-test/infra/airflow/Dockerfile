FROM apache/airflow:3.1.7-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
RUN usermod -aG docker airflow
USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-docker>=4.0.0 \
    apache-airflow-providers-standard>=0.0.1
