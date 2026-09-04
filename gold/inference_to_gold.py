import os
import time
import tempfile
import json
import psycopg2
import requests
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, current_timestamp, pandas_udf
from pyspark.sql.types import DoubleType
from delta import configure_spark_with_delta_pip
from datetime import datetime

# Configuration
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

DB_CONFIG = {
    "host": "postgres",
    "database": "fraud_db",
    "user": "fraud_user",
    "password": os.getenv("DB_PASSWORD"),
    "port": 5432
}

def safe_cast(val, to_type):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return None

# Slack Alerts configurations
def send_slack_alerts(high_risk_pdf):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("Warning: SLACK_WEBHOOK_URL not set. Skipping alerts.")
        return

    if high_risk_pdf.empty:
        print("No high-risk transactions found. No alerts sent.")
        return

    print(f"Sending {len(high_risk_pdf)} Slack alerts...")

    for _, row in high_risk_pdf.iterrows():
        message = f"FRAUD ALERT: Transaction {row['transaction_id'][:8]}... | Amount: ${row['transaction_amount']:.2f} | Score: {row['xgboost_probability']:.4f}"
        try:
            response = requests.post(
                webhook_url,
                data=json.dumps({"text": message}),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send alert: {e}")
        time.sleep(1)


# Upstream User Dimension Sync Pattern
def sync_dim_users_to_postgres(spark):
    print("Syncing user dimensions to PostgreSQL...")
    silver_user_path = "s3a://fraud-detection-lake-nouman-v2/silver/dim_user/"
    user_df = spark.read.format("delta").load(silver_user_path)
    
    db_url = "jdbc:postgresql://postgres:5432/fraud_db"
    db_properties = {
        "user": "fraud_user",
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }
    
    user_df.write \
        .mode("overwrite") \
        .option("batchsize", "10000") \
        .jdbc(db_url, "dim_user_staging", properties=db_properties)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dim_user (user_id)
        SELECT DISTINCT user_id FROM dim_user_staging
        ON CONFLICT (user_id) DO NOTHING;
    """)
    conn.commit()
    
    cursor.execute("DROP TABLE IF EXISTS dim_user_staging;")
    conn.commit()
    
    cursor.close()
    conn.close()
    print("User dimensions synchronized successfully.")


# Upstream Merchant Dimension Sync Pattern
def sync_dim_merchants_to_postgres(spark):
    print("Syncing merchant dimensions to PostgreSQL...")
    silver_merchant_path = "s3a://fraud-detection-lake-nouman-v2/silver/dim_merchant/"
    
    try:
        merchant_df = spark.read.format("delta").load(silver_merchant_path)
    except Exception:
        silver_fact_path = "s3a://fraud-detection-lake-nouman-v2/silver/fact_fraud_inference/"
        fact_df = spark.read.format("delta").load(silver_fact_path)
        merchant_df = fact_df.select("merchant_id").distinct()

    db_url = "jdbc:postgresql://postgres:5432/fraud_db"
    db_properties = {
        "user": "fraud_user",
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }
    
    merchant_df.write \
        .mode("overwrite") \
        .option("batchsize", "10000") \
        .jdbc(db_url, "dim_merchant_staging", properties=db_properties)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dim_merchant (merchant_id)
        SELECT DISTINCT merchant_id FROM dim_merchant_staging
        ON CONFLICT (merchant_id) DO NOTHING;
    """)
    conn.commit()
    
    cursor.execute("DROP TABLE IF EXISTS dim_merchant_staging;")
    conn.commit()
    
    cursor.close()
    conn.close()
    print("Merchant dimensions synchronized successfully.")


# Upstream Time Dimension Sync Pattern
def sync_dim_time_to_postgres(spark):
    print("Syncing time dimensions to PostgreSQL...")
    silver_time_path = "s3a://fraud-detection-lake-nouman-v2/silver/dim_time/"
    
    try:
        time_df = spark.read.format("delta").load(silver_time_path)
    except Exception:
        silver_fact_path = "s3a://fraud-detection-lake-nouman-v2/silver/fact_fraud_inference/"
        fact_df = spark.read.format("delta").load(silver_fact_path)
        time_df = fact_df.select("time_id").distinct()

    db_url = "jdbc:postgresql://postgres:5432/fraud_db"
    db_properties = {
        "user": "fraud_user",
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }
    
    time_df.write \
        .mode("overwrite") \
        .option("batchsize", "10000") \
        .jdbc(db_url, "dim_time_staging", properties=db_properties)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dim_time (time_id)
        SELECT DISTINCT time_id FROM dim_time_staging
        ON CONFLICT (time_id) DO NOTHING;
    """)
    conn.commit()
    
    cursor.execute("DROP TABLE IF EXISTS dim_time_staging;")
    conn.commit()
    
    cursor.close()
    conn.close()
    print("Time dimensions synchronized successfully.")


# fetch dim_model metadata & UPSERTS into dim_model from MLflow
def populate_dim_model(client, model_version_obj, run_id):
    print("Populating dim_model from MLflow registry...")

    run = client.get_run(run_id)
    auc_score = run.data.metrics.get("auc", 0.0)
    
    n_estimators = safe_cast(run.data.params.get("n_estimators"), int)
    max_depth = safe_cast(run.data.params.get("max_depth"), int)
    learning_rate = safe_cast(run.data.params.get("learning_rate"), float)
    decision_threshold = 0.5 

    model_id = f"{model_version_obj.name}_v{model_version_obj.version}"

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO dim_model (
            model_id, mlflow_run_id, model_name, model_version,
            stage, auc_score, decision_threshold,
            n_estimators, max_depth, learning_rate, registered_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_id) DO UPDATE SET
            auc_score = EXCLUDED.auc_score,
            stage = EXCLUDED.stage,
            registered_at = EXCLUDED.registered_at;
    """, (
        model_id,
        run_id,
        model_version_obj.name,
        model_version_obj.version,
        "Production",
        auc_score,
        decision_threshold,
        n_estimators,
        max_depth,
        learning_rate,
        datetime.utcnow()
    ))

    cursor.close()
    conn.close()
    print(f"dim_model populated: {model_id} | AUC: {auc_score:.4f}")
    return model_id


# Scalable Upsert to PostgreSQL fact_fraud_inference using Spark JDBC batch writes
def upsert_to_postgres(scored_spark_df, spark):
    print("Writing to PostgreSQL via distributed staging + upsert pattern...")

    db_url = "jdbc:postgresql://postgres:5432/fraud_db"
    db_properties = {
        "user": "fraud_user",
        "password": os.getenv("DB_PASSWORD"),
        "driver": "org.postgresql.Driver"
    }

    scored_spark_df.write \
        .mode("overwrite") \
        .option("batchsize", "10000") \
        .jdbc(db_url, "fact_fraud_inference_staging", properties=db_properties)
    
    print("Staging table written.")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO fact_fraud_inference (
            transaction_id, user_id, merchant_id, time_id, model_id,
            transaction_amount, xgboost_probability, is_fraud,
            inference_latency_ms, inference_timestamp
        )
        SELECT
            transaction_id, user_id, merchant_id, time_id, model_id,
            transaction_amount, xgboost_probability, is_fraud,
            inference_latency_ms, inference_timestamp
        FROM fact_fraud_inference_staging
        ON CONFLICT (transaction_id) DO UPDATE SET
            xgboost_probability = EXCLUDED.xgboost_probability,
            is_fraud = EXCLUDED.is_fraud,
            model_id = EXCLUDED.model_id,
            inference_latency_ms = EXCLUDED.inference_latency_ms,
            inference_timestamp = EXCLUDED.inference_timestamp;
    """)

    rows_affected = cursor.rowcount
    conn.commit()

    cursor.execute("DROP TABLE IF EXISTS fact_fraud_inference_staging;")
    conn.commit()

    cursor.close()
    conn.close()
    print(f"Upsert complete. {rows_affected} rows affected in fact_fraud_inference.")


def run_batch_inference():
    print("=" * 50)
    print("Starting Distributed Gold Layer Batch Inference Pipeline")
    print("=" * 50)

    print("\n[1/5] Loading Production model from MLflow registry...")
    client = MlflowClient()
    model_name = "fraud_detection_xgboost"

    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        print("ERROR: No Production model found in MLflow registry. Exiting.")
        return

    model_version_obj = versions[0]
    run_id = model_version_obj.run_id
    print(f"Found Production model: version={model_version_obj.version}, run_id={run_id[:8]}...")

    model = xgb.XGBClassifier()
    with tempfile.TemporaryDirectory() as tmp_dir:
        client.download_artifacts(run_id, "xgboost_fraud_model/model.json", tmp_dir)
        model.load_model(os.path.join(tmp_dir, "xgboost_fraud_model", "model.json"))
    print("Model loaded successfully.")

    print("\n[2/5] Populating dim_model...")
    model_id = populate_dim_model(client, model_version_obj, run_id)

    print("\n[3/5] Initializing Spark session...")
    builder = SparkSession.builder \
        .appName("GoldBatchInferenceDistributed") \
        .master("local[*]") \
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.1.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                "org.postgresql:postgresql:42.5.4") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"]) \
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"]) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Sync all dimensions upstream before processing facts
    sync_dim_users_to_postgres(spark)
    sync_dim_merchants_to_postgres(spark)
    sync_dim_time_to_postgres(spark)

    print("\n[4/5] Reading Silver layer...")
    silver_path = "s3a://fraud-detection-lake-nouman-v2/silver/fact_fraud_inference/"
    df = spark.read.format("delta").load(silver_path)
    
    if "amount" in df.columns and "transaction_amount" not in df.columns:
        df = df.withColumnRenamed("amount", "transaction_amount")

    feature_cols = [
        "transaction_amount", 
        "oldbalanceOrg", 
        "newbalanceOrig", 
        "is_balance_fraud_signal"
    ]

    print("Running Distributed XGBoost inference via Pandas UDF...")
    start_time = time.time()

    broadcast_model = spark.sparkContext.broadcast(model)

    @pandas_udf(DoubleType())
    def predict_fraud_udf(c1: pd.Series, c2: pd.Series, c3: pd.Series, c4: pd.Series) -> pd.Series:
        X_worker = pd.concat([c1, c2, c3, c4], axis=1)
        X_worker.columns = feature_cols
        model_inst = broadcast_model.value
        probs = model_inst.predict_proba(X_worker)[:, 1]
        return pd.Series(probs)

    scored_spark_df = df.withColumn(
        "xgboost_probability", 
        predict_fraud_udf(
            col("transaction_amount"),
            col("oldbalanceOrg"),
            col("newbalanceOrig"),
            col("is_balance_fraud_signal")
        )
    ).withColumn(
        "is_fraud", 
        when(col("xgboost_probability") > 0.5, 1).otherwise(0)
    )

    latency_ms = (time.time() - start_time) * 1000

    scored_spark_df = scored_spark_df \
        .withColumn("model_id", lit(model_id)) \
        .withColumn("inference_latency_ms", lit(latency_ms)) \
        .withColumn("inference_timestamp", current_timestamp())

    cols_to_write = [
        "transaction_id", "user_id", "merchant_id", "time_id", "model_id",
        "transaction_amount", "xgboost_probability", "is_fraud",
        "inference_latency_ms", "inference_timestamp"
    ]

    final_scored_spark_df = scored_spark_df.select([col(c) for c in cols_to_write])

    print("\n[5/5] Upserting to PostgreSQL fact_fraud_inference...")
    upsert_to_postgres(final_scored_spark_df, spark)

    high_risk_pdf = final_scored_spark_df \
        .filter(col("xgboost_probability") > 0.9) \
        .orderBy(col("xgboost_probability").desc()) \
        .limit(10) \
        .toPandas()
    
    send_slack_alerts(high_risk_pdf)

    print("\n" + "=" * 50)
    print("Distributed Gold Layer Inference Pipeline Complete!")
    print("=" * 50)

    spark.stop()


if __name__ == "__main__":
    run_batch_inference()