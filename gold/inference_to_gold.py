import os
import tempfile
import pandas as pd
import xgboost as xgb
import mlflow
import requests
import json
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# 1. Point to the central MLflow tracking server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

def send_slack_alerts(dataframe):
    # Filters for high fraud scores and sends a message to Slack.
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("Warning: SLACK_WEBHOOK_URL environment variable not set. Skipping alerts.")
        return

    # Filter for transactions scoring > 0.9
    high_risk_df = dataframe[dataframe["fraud_score"] > 0.9]
    
    if high_risk_df.empty:
        print("No high-risk transactions found in this batch. No alerts sent.")
        return

    print(f"Triggering {len(high_risk_df)} Slack alerts for high-risk transactions...")
    
    for _, row in high_risk_df.iterrows():
        # message format tahat will ve sent to slack
        message = f"ALERT: Transaction {row['transaction_id']}, Amount ${row['amount']:.2f}, Score: {row['fraud_score']:.4f}"
        payload = {"text": message}
        
        try:
            response = requests.post(
                webhook_url, 
                data=json.dumps(payload), 
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send alert for {row['transaction_id']}: {e}")

def run_batch_inference():
    print("Connecting to MLflow Registry...")
    
    # 2. Define the model name
    model_name = "fraud_detection_xgboost"
    
    client = MlflowClient()
    
    # Get latest Production model version
    versions = client.get_latest_versions(model_name, stages=["Production"])
    
    if not versions:
        print("No Production model found in the registry.")
        return
        
    run_id = versions[0].run_id
    
    model = xgb.XGBClassifier()
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Downloading model artifact from MLflow...")
        client.download_artifacts(run_id, "xgboost_fraud_model/model.json", tmp_dir)
        model.load_model(os.path.join(tmp_dir, "xgboost_fraud_model", "model.json"))
        
    print("Successfully loaded Production model!")

    # 3. Initialize Spark 
    builder = SparkSession.builder \
        .appName("GoldBatchInference") \
        .master("local[2]") \
        .config("spark.jars.packages", 
                "io.delta:delta-spark_2.12:3.1.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                "org.postgresql:postgresql:42.5.4") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    # 4. Read from Silver layer
    silver_data_path = "s3a://fraud-detection-lake-nouman-v2/silver/fact_transactions/"
    print(f"Reading Silver transactions from: {silver_data_path}")
    df = spark.read.format("delta").load(silver_data_path)
    
    pdf = df.toPandas()
    
    # 5. Generate Fraud Probability Scores
    feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "is_balance_fraud_signal"]
    X_infer = pdf[feature_cols]

    print("Running model inference...")
    pdf["fraud_score"] = model.predict_proba(X_infer)[:, 1]
    pdf["fraud_flag"] = pdf["fraud_score"] > 0.5
    
    cols_to_write = [
        "transaction_id", "timestamp", "customer_id", "merchant_id", 
        "amount", "fraud_score", "fraud_flag"
    ]
    final_pdf = pdf[cols_to_write]

    scored_df = spark.createDataFrame(final_pdf)

    # 6. Write to PostgreSQL
    print("Writing scored data to PostgreSQL...")
    db_url = "jdbc:postgresql://localhost:5432/fraud_db"
    db_properties = {
        "user": "fraud_user",
        "password": "fraud_pass",
        "driver": "org.postgresql.Driver"
    }
    
    scored_df.write.jdbc(
        url=db_url,
        table="fraud_scores",
        mode="append",
        properties=db_properties
    )
    
    # 7. Fire Slack Alerts
    send_slack_alerts(final_pdf)
    
    print("Batch inference, PostgreSQL load, and alerting completed successfully!")
    spark.stop()

if __name__ == "__main__":
    run_batch_inference()