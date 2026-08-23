import os
import tempfile
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


# Step_01 : Load Production_Model
def run_batch_inference():
    print("Connecting to MLflow Registry...")
    
    # Define the model name
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
        
        # Load the downloaded file into the model instance
        model.load_model(os.path.join(tmp_dir, "xgboost_fraud_model", "model.json"))
        
    print("Successfully loaded Production model!")

    # 3. Initialize Spark (PostgreSQL JDBC Driver are added)
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

    # 4. Read from Silver layer (fact_transactions)
    silver_data_path = "s3a://fraud-detection-lake-nouman-v2/silver/fact_transactions/"
    print(f"Reading Silver transactions from: {silver_data_path}")
    df = spark.read.format("delta").load(silver_data_path)

    # Convert to Pandas for XGBoost inference
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
    

    # 6. Write to PostgreSQL (The Serving Layer)
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
    
    print("Batch inference and PostgreSQL load completed successfully!")
    spark.stop()

if __name__ == "__main__":
    run_batch_inference()









