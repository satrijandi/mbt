# Telecom Churn Prediction Example

This is an example MBT project demonstrating churn prediction for a telecom company.

## Quick Start

```bash
# Install mbt-core in development mode
pip install -e ../../mbt-core

# Compile the pipeline
mbt compile churn_simple_v1

# Run the pipeline
mbt run --select churn_simple_v1

# Validate pipelines
mbt validate
```

## Pipeline

The `churn_simple_v1` pipeline demonstrates a basic training workflow:
1. Load customer data from CSV
2. Split into train/test sets (80/20)
3. Train a RandomForest classifier
4. Evaluate on test set

## Data

Sample data in `sample_data/customers.csv` contains:
- customer_id: Unique identifier
- tenure: Months as customer
- monthly_charges: Monthly bill amount
- total_charges: Total amount charged
- churned: 1 if customer left, 0 if stayed (target variable)
