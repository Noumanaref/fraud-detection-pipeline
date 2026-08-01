import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# 1. Force MLflow to point to the central tracking server and shared volume artifact path
mlflow.set_tracking_uri("http://mlflow:5000")




def train_fraud_model():
    # 1. Initialize Spark Session with Delta Lake and Cluster support
    builder = SparkSession.builder \
        .appName("FraudModelTraining") \
        .master("spark://spark-master:7077") \
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.1.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    print("Spark Session initialized successfully for Model Training!")

    # 2. Read the Gold Layer Feature Store Table from S3
    gold_feature_path = "s3a://fraud-detection-lake-nouman/gold/ml_features/"
    print(f"Reading Gold features from: {gold_feature_path}")
    feature_df = spark.read.format("delta").load(gold_feature_path)

    # 3. Convert Spark DataFrame to Pandas for XGBoost training
    print("Converting feature DataFrame to Pandas...")
    pdf = feature_df.toPandas()

    # Sort data by timestamp to ensure a correct time-based split
    pdf = pdf.sort_values("timestamp").reset_index(drop=True)

    # 4. Define feature columns and target label
    feature_cols = [
        "amount", 
        "oldbalanceOrg", 
        "newbalanceOrig", 
        "is_balance_fraud_signal"
    ]
    target_col = "isFraud"

    X = pdf[feature_cols]
    y = pdf[target_col]

    # 5. Time-based Split: 80% older records for training, 20% newer for testing
    split_index = int(len(pdf) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]    
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:] 

    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # 6. Configure MLflow Experiment Tracking & Training
    mlflow.set_experiment("fraud_detection_xgboost")

    with mlflow.start_run() as run:
        # Define Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "scale_pos_weight": 773,  # Fix for the 773:1 class imbalance ratio
            "eval_metric": "logloss",
            "random_state": 42
        }

        # Log parameters to MLflow
        mlflow.log_params(params)

        print("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        print("Feature Importances:")
        for col, imp in zip(feature_cols, model.feature_importances_):
            print(f"  {col}: {imp:.4f}")

        # Predict probabilities on test set
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Calculate Evaluation Metrics
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)

        print(f"Model Evaluation Metrics -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        # Log trained model artifact to MLflow
        mlflow.xgboost.log_model(model, "xgboost_model")
        print("Model successfully logged to MLflow!")

    spark.stop()

if __name__ == "__main__":
    train_fraud_model()