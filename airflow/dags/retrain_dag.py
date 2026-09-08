from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

SPARK_CONTAINER = "fraud-detection-pipeline-spark-master-1"

SPARK_SUBMIT = (
    f"docker exec "
    f"-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID "
    f"-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY "
    f"-e MLFLOW_TRACKING_URI=http://mlflow:5000 "
    f"-e DB_PASSWORD=$DB_PASSWORD "
    f"{SPARK_CONTAINER} "
    f"/opt/spark/bin/spark-submit "
    f"--packages 'io.delta:delta-spark_2.12:3.1.0,"
    f"org.apache.hadoop:hadoop-aws:3.3.4,"
    f"com.amazonaws:aws-java-sdk-bundle:1.12.262,"
    f"org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0' "
    f"--conf 'spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension' "
    f"--conf 'spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog' "
    f"--conf 'spark.hadoop.fs.s3a.access.key=$AWS_ACCESS_KEY_ID' "
    f"--conf 'spark.hadoop.fs.s3a.secret.key=$AWS_SECRET_ACCESS_KEY' "
    f"--conf 'spark.hadoop.fs.s3a.endpoint=s3.amazonaws.com' "
)

with DAG(
    dag_id="retrain_dag",
    default_args=DEFAULT_ARGS,
    description="Weekly model retraining and evaluation pipeline",
    schedule_interval="0 2 * * 0",
    start_date=datetime(2026, 9, 1),
    max_active_runs = 1,
    catchup=False,
    tags=["fraud", "mlops", "retraining"],
) as dag:

    # Task 1: Fetch training data
    fetch_training_data = BashOperator(
        task_id="fetch_training_data",
        bash_command="echo 'Pulling last 30 days from Silver Delta lake...'",
    )

    # Task 2: Run Feature Engineering
    run_feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command=SPARK_SUBMIT + "/workspace/ml/feature_engineering.py",
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.aws_access_key_id }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.aws_secret_access_key }}",
            "DB_PASSWORD": "{{ var.value.db_password }}",
        }
    )

    # Task 3: Train New Model
    train_new_model = BashOperator(
        task_id="train_new_model",
        bash_command="python /workspace/ml/model_train.py",
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.aws_access_key_id }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.aws_secret_access_key }}",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "DB_PASSWORD": "{{ var.value.db_password }}",
        }
    )

    # Task 4: Evaluate and Promote
    evaluate_and_promote = BashOperator(
        task_id="evaluate_and_promote",
        bash_command="python /workspace/ml/evaluate_model.py",
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.aws_access_key_id }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.aws_secret_access_key }}",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "DB_PASSWORD": "{{ var.value.db_password }}",
        }
    )

    # Task 5: Notify Result
    notify_result = BashOperator(
        task_id="notify_result",
        bash_command="echo 'Model retrained and evaluated successfully.'",
    )

    # Sequence dependencies: 1 → 2 → 3 → 4 → 5
    fetch_training_data >> run_feature_engineering >> train_new_model >> evaluate_and_promote >> notify_result