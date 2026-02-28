-- ==========================================================================
-- MBT Integration Test - PostgreSQL Initialization
-- Creates databases and users only. Warehouse tables and seed data are
-- loaded separately by 02-init-data.sh.
-- ==========================================================================

-- MLflow backend database
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- Airflow backend database
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Warehouse database (ML pipeline data)
CREATE DATABASE warehouse;
CREATE USER mbt_user WITH PASSWORD 'mbt_password';
GRANT ALL PRIVILEGES ON DATABASE warehouse TO mbt_user;

-- Gitea backend database
CREATE DATABASE gitea_db;
CREATE USER gitea_user WITH PASSWORD 'gitea_password';
GRANT ALL PRIVILEGES ON DATABASE gitea_db TO gitea_user;

-- Metabase backend database
CREATE DATABASE metabase_db;
CREATE USER metabase_user WITH PASSWORD 'metabase_password';
GRANT ALL PRIVILEGES ON DATABASE metabase_db TO metabase_user;

-- Allow metabase to connect to warehouse for BI dashboards
GRANT CONNECT ON DATABASE warehouse TO metabase_user;

-- Connect to mlflow_db and grant schema permissions
\connect mlflow_db;
GRANT ALL ON SCHEMA public TO mlflow_user;

-- Connect to airflow_db and grant schema permissions
\connect airflow_db;
GRANT ALL ON SCHEMA public TO airflow_user;

-- Connect to gitea_db and grant schema permissions
\connect gitea_db;
GRANT ALL ON SCHEMA public TO gitea_user;

-- Connect to metabase_db and grant schema permissions
\connect metabase_db;
GRANT ALL ON SCHEMA public TO metabase_user;

-- Connect to warehouse and grant schema permissions
\connect warehouse;
GRANT ALL ON SCHEMA public TO mbt_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mbt_user;

-- Allow metabase to read warehouse tables for BI dashboards
GRANT USAGE ON SCHEMA public TO metabase_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO metabase_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO metabase_user;
