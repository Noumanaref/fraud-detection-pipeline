# this is standard librray to interact with OS, access enviornment variables like (AWS credentials, mlflow URL) 
import os
# tempfile will create secure, temporary directories in our local machine and will automatically discard it when done.
import tempfile
# mlflow library for tracking experiements, managing runs and logging OR fetching artifacts.
import mlflow
import xgboost as xgb
# mlflow client : A low-level Python API client for MLflow that allows querying experiments,
# searching runs, downloading raw artifacts, and controlling the Model Registry (e.g., promoting models to "Production").
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
# A helper function provided by Delta Lake that configures Spark with the necessary Delta Lake JAR dependencies and extensions
from delta import configure_spark_with_delta_pip
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# configuration and initialization
# This basically tells the mlflow client where the mlflow server is located
# Checks if the environment variable MLFLOW_TRACKING_URI exists; if not, falls back to http://mlflow:5000.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


# initialize the Spark Session
def evaluate_and_promote_model():
    builder = SparkSession.builder \
        .appName("FraudModelEvaluation") \
        .master("local[2]") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    # APPname : basically names the spark application (FraudModelEvaluation) for monitoring and logging purposes.
    # master("local[2]") : configure the spark to run locally with 2 CPU threads. This is useful for development and testing.
    # configure_spark_with_delta_pip(builder).getOrCreate(): Merges Delta dependencies into the builder and spins up the Spark session.

    # STEP _ 2: Fetching and splitting feature data
    print("Fetching Gold features for evaluation...")
    gold_feature_path = "s3a://fraud-detection-lake-nouman-v2/gold/ml_features/"
    feature_df = spark.read.format("delta").load(gold_feature_path)

    pdf = feature_df.toPandas()
    pdf = pdf.sort_values("timestamp").reset_index(drop=True)

    feature_cols = ["transaction_amount", "oldbalanceOrg", "newbalanceOrig", "is_balance_fraud_signal"]
    target_col = "isFraud"

    X = pdf[feature_cols]
    y = pdf[target_col]

    split_index = int(len(pdf) * 0.80)
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]


    # STEP _ 3: Finding the Latest MLflow Run
    client = MlflowClient()
    experiment_name = "fraud_detection_xgboost_v2"

    artifact_path = "xgboost_fraud_model"
    registry_model_name = "fraud_detection_xgboost"

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found.")
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    auc_score = latest_run.data.metrics.get("auc", 0)
    print(f"\nLoaded Run ID: {run_id} | Logged AUC: {auc_score:.4f}")


    # STEP _ 4: Downloading the Model Artifact
    model = xgb.XGBClassifier()
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Downloading model artifact from MLflow...")
        client.download_artifacts(run_id, f"{artifact_path}/model.json", tmp_dir)
        local_model_path = os.path.join(tmp_dir, artifact_path, "model.json")
        model.load_model(local_model_path)


    # STEP _ 5: Evaluating the Model
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Extended Evaluation Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # STEP _ 6: Model Promotion Logic
    if auc_score > 0.85:
        print("\nModel meets the 0.85 AUC threshold. Registering to Model Registry...")

        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri, registry_model_name)
        print(f"Promoting '{registry_model_name}' to 'Production' stage...")

        client.transition_model_version_stage(name=registry_model_name,version=mv.version,stage="Production",archive_existing_versions=True)
        print("Success! Model is ready for the Gold layer inference pipeline.")
    else:
        print("\nModel rejected. AUC is below the 0.85 threshold. Tuning required.")
    spark.stop()

if __name__ == "__main__":
    evaluate_and_promote_model()







