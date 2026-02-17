How to see it in seaweefs ui
http://localhost:8888/buckets/mbt-mlflow-artifacts/1/15d892a66c2b48988b187800e2dcc5a4/artifacts/model/

Migrate to latest airflow version for the integration-test
I want to upgrade the airflow in the integration-test using apache/airflow:3.1.7 Please check whether mbt-airflow needs to be updated. If so please update it accordingly, I want to see the integration test still works perfectly in the end.

Migrate to latest mlflow version v3.9.0
I want to upgrade the mlflow in the integration-test using ghcr.io/mlflow/mlflow:v3.9.0 Please check whether mbt-mlflow needs to be updated. If so please update it accordingly, I want to see the integration test still works perfectly in the end.

Migrate to latest postgres version v18.2.1
I want to upgrade the postgres in the integration-test using postgres:v18.2.1 Please check whether mbt-postgres needs to be updated. If so please update it accordingly, I want to see the integration test still works perfectly in the end.



CI/CD pipeline example (Zot registry/Woodpecker CI/Gitea)

H2O server integration
Add mbt-snowflake same like mbt-postgres
See airflow-dag manifest in Gitea