import os
import mlflow
import mlflow.xgboost
import tempfile
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.sql.functions import col as spark_col
from delta import configure_spark_with_delta_pip
from sklearn.metrics import roc_auc_score, precision_score

# 1. Force MLflow to point to the central tracking server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


def train_fraud_model():
    # 1. Initialize Spark Session with optimized cluster settings matching Silver/Gold pipelines
    builder = (
        SparkSession.builder.appName("FraudModelTraining")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.1.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("Spark Session initialized successfully with optimized cluster support!")

    # 2. Read the Gold Layer Feature Store Table from S3
    gold_feature_path = "s3a://fraud-detection-lake-nouman-v2/gold/ml_features/"
    print(f"Reading Gold features from: {gold_feature_path}")
    feature_df = spark.read.format("delta").load(gold_feature_path)

    # Fix 1: Sample ~15% (approx 500K rows out of 3.5M) to prevent driver OOM during Pandas conversion
    feature_df = feature_df.sample(fraction=0.15, seed=42)
    print(f"Sampled dataset size: {feature_df.count()} rows")

    # 3. Optimize memory before converting to Pandas by selecting required columns and casting types
    print("Selecting model features and casting to optimal types...")
    feature_cols = [
        "transaction_amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "is_balance_fraud_signal",
    ]
    target_col = "isFraud"

    # Fix 2: Cast boolean/other types properly to integer before collection
    optimized_df = feature_df.select(
        ["timestamp"] + feature_cols + [target_col]
    ).withColumn(
        "is_balance_fraud_signal",
        spark_col("is_balance_fraud_signal").cast("integer"),
    )

    print("Converting optimized feature DataFrame to Pandas...")
    pdf = optimized_df.toPandas()

    # Sort data by timestamp to ensure a correct time-based split
    pdf = pdf.sort_values("timestamp").reset_index(drop=True)

    # Drop timestamp column now that sorting is complete so it's not passed to XGBoost
    pdf = pdf.drop(columns=["timestamp"])

    X = pdf[feature_cols]
    y = pdf[target_col]

    # 5. Time-based Split: 80% older records for training, 20% newer for testing
    split_index = int(len(pdf) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # 6. Configure MLflow Experiment Tracking & Training
    mlflow.set_experiment("fraud_detection_xgboost_v2")

    # Define Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": 773,  # Fix for the 773:1 class imbalance ratio
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,  # Utilize all available CPU cores for XGBoost fitting
    }

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(**params)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", precision)

        # Ensure local models directory exists
        os.makedirs("/workspace/models", exist_ok=True)
        # Save a unique versioned copy using the run ID
        versioned_model_path = f"/workspace/models/xgboost_model_{run.info.run_id}.pkl"


        model.save_model(versioned_model_path)

        # Also save a standard pointer copy for pipeline consistency
        latest_model_path = "/workspace/models/xgboost_model.pkl"
        model.save_model(latest_model_path)

        print(f"Model saved locally at : {versioned_model_path} and {latest_model_path}")

        # Log model artifact to MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.json")
            model.save_model(model_path)
            mlflow.log_artifact(model_path, artifact_path="xgboost_fraud_model")

        print(f"Run ID: {run.info.run_id}")
        print(f"AUC: {auc:.4f} | Precision: {precision:.4f}")

    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/xgboost_fraud_model",
        name="fraud_detection_xgboost",
    )

    print(f"Training_Completed_Successfully! Model registered in MLflow Model Registry & at {versioned_model_path}.")

    spark.stop()


if __name__ == "__main__":
    train_fraud_model()