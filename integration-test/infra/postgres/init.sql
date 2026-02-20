-- MLflow backend database
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- Airflow backend database
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Warehouse database
CREATE DATABASE warehouse;
CREATE USER mbt_user WITH PASSWORD 'mbt_password';
GRANT ALL PRIVILEGES ON DATABASE warehouse TO mbt_user;

-- Connect to mlflow_db and grant schema permissions
\connect mlflow_db;
GRANT ALL ON SCHEMA public TO mlflow_user;

-- Connect to airflow_db and grant schema permissions
\connect airflow_db;
GRANT ALL ON SCHEMA public TO airflow_user;

-- Connect to warehouse and grant schema permissions
\connect warehouse;
GRANT ALL ON SCHEMA public TO mbt_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mbt_user;

-- Feature + label table (seeded with historical data)
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    tenure INT NOT NULL,
    monthly_charges FLOAT NOT NULL,
    total_charges FLOAT NOT NULL,
    churned INT NOT NULL CHECK (churned IN (0, 1))
);

-- Scoring table (new customers without labels)
CREATE TABLE customers_to_score (
    customer_id VARCHAR(20) PRIMARY KEY,
    tenure INT NOT NULL,
    monthly_charges FLOAT NOT NULL,
    total_charges FLOAT NOT NULL
);

-- Prediction output table (written by serving pipeline)
CREATE TABLE churn_predictions (
    customer_id VARCHAR(20),
    prediction INT,
    prediction_probability FLOAT,
    execution_date TIMESTAMP DEFAULT NOW(),
    model_run_id VARCHAR(100),
    serving_run_id VARCHAR(100)
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mbt_user;

-- Seed historical customer data
INSERT INTO customers (customer_id, tenure, monthly_charges, total_charges, churned) VALUES
('CUST_00001', 12, 45.50, 546.00, 0),
('CUST_00002', 36, 89.20, 3211.20, 1),
('CUST_00003', 8, 35.75, 286.00, 0),
('CUST_00004', 48, 102.40, 4915.20, 0),
('CUST_00005', 3, 55.00, 165.00, 1),
('CUST_00006', 24, 78.90, 1893.60, 0),
('CUST_00007', 6, 42.30, 253.80, 1),
('CUST_00008', 60, 95.00, 5700.00, 0),
('CUST_00009', 1, 29.99, 29.99, 1),
('CUST_00010', 18, 65.50, 1179.00, 0),
('CUST_00011', 30, 88.00, 2640.00, 0),
('CUST_00012', 2, 50.00, 100.00, 1),
('CUST_00013', 42, 110.00, 4620.00, 0),
('CUST_00014', 5, 38.50, 192.50, 1),
('CUST_00015', 15, 72.00, 1080.00, 0),
('CUST_00016', 9, 60.00, 540.00, 1),
('CUST_00017', 36, 85.00, 3060.00, 0),
('CUST_00018', 4, 45.00, 180.00, 1),
('CUST_00019', 24, 92.00, 2208.00, 0),
('CUST_00020', 7, 33.00, 231.00, 1);

-- Seed scoring data (new customers without labels)
INSERT INTO customers_to_score (customer_id, tenure, monthly_charges, total_charges) VALUES
('CUST_NEW_001', 12, 65.50, 786.00),
('CUST_NEW_002', 3, 45.20, 135.60),
('CUST_NEW_003', 24, 89.30, 2143.20),
('CUST_NEW_004', 6, 52.70, 316.20),
('CUST_NEW_005', 18, 75.00, 1350.00),
('CUST_NEW_006', 1, 30.00, 30.00),
('CUST_NEW_007', 36, 95.50, 3438.00),
('CUST_NEW_008', 9, 55.00, 495.00),
('CUST_NEW_009', 48, 105.00, 5040.00),
('CUST_NEW_010', 2, 40.00, 80.00);
