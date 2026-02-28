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

Shifting features is part of feature selection process not just evaluation


Jupyterhub login invalid
For metabase is it possible to setup the user without setup on first access
Add pgadmin with existing connection to all the db.


Please update @postgres/init.sql where it will
Init data in postgres but accessible to also via metabase
customers to score : contains customer_id, snapshot_date
features table a: contains customer_id, snapshot_date, feature_a, …, feature_z
features table b: contains customer_id, snapshot_date, feature_1, …, feature_1000
label contains customer_id, snapshot-date, is_churn
makes the data as realistic as it can be. let’s say snapshot_date is the cutoff date / prediction date and the label is_churn meaning whether the customer will churn in the next 1 months.
Test your code before done and provide the report.


Please provide a bash script 03-mbt-init.sh simulating DS to do mbt init, create churn_training_pipeline_v1 yaml and after run that run the pipeline in jupyter notebook
DS then create MR to DE
CI runs
DE approve and merge to main branch : which also means will deploy the RAG to airflow eventually in the end. skip the testing on dev staging prod
DS the can run the pipeline on airflow
DS then create a serving pipeline churn_serving_pipeline_v1.yaml
Run the pipeline in the jupyter notebook
DS then create MR to DE
CI runs
DE approve and merge to main branch : which also means will deploy the RAG to airflow eventually in the end. skip the testing on dev staging prod. Since the changes in the repo is adding the serving pipeline, the training pipeline should not change. If the changes in on existing pipeline, then it cannot be done.
DS the can run the pipeline on airflow
