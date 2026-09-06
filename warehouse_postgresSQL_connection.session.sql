SELECT *
FROM fact_fraud_inference 
LIMIT 5;


-- 1. Dimension: User (SCD Type 2)
CREATE TABLE IF NOT EXISTS dim_user (
    user_id VARCHAR(64) PRIMARY KEY,
    customer_name VARCHAR(255),
    risk_tier VARCHAR(50),
    registered_location VARCHAR(255),
    account_age_days INT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN
);

select * from dim_user;

-- 2. Dimension: Merchant (SCD Type 1)
CREATE TABLE IF NOT EXISTS dim_merchant (
    merchant_id VARCHAR(64) PRIMARY KEY,
    merchant_name VARCHAR(255),
    merchant_category VARCHAR(100),
    terminal_location VARCHAR(255),
    channel VARCHAR(50)
);

select * from dim_merchant;

-- 3. Dimension: Time (Conformed)
CREATE TABLE IF NOT EXISTS dim_time (
    time_id VARCHAR(64) PRIMARY KEY,
    full_timestamp TIMESTAMP,
    hour INT,
    day INT,
    month INT,
    year INT,
    day_of_week VARCHAR(20),
    is_weekend BOOLEAN
);

-- 4. Dimension: Model (MLOps Context)
CREATE TABLE IF NOT EXISTS dim_model (
    model_id VARCHAR(64) PRIMARY KEY,
    mlflow_run_id VARCHAR(255),
    model_name VARCHAR(255),
    model_version VARCHAR(50),
    stage VARCHAR(50),
    auc_score DOUBLE PRECISION,
    decision_threshold DOUBLE PRECISION,
    n_estimators INT,
    max_depth INT,
    learning_rate DOUBLE PRECISION,
    registered_at TIMESTAMP
);

-- 5. Fact: Fraud Inference
CREATE TABLE IF NOT EXISTS fact_fraud_inference (
    transaction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES dim_user(user_id),
    merchant_id VARCHAR(64) REFERENCES dim_merchant(merchant_id),
    time_id VARCHAR(64) REFERENCES dim_time(time_id),
    model_id VARCHAR(64) REFERENCES dim_model(model_id),
    transaction_amount NUMERIC(15, 2),
    xgboost_probability DOUBLE PRECISION,
    is_fraud INT,
    inference_latency_ms DOUBLE PRECISION,
    inference_timestamp TIMESTAMP
);

select * from fact_fraud_inference;


