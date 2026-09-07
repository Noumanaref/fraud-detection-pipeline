from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
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

SPARK_SUBMIT_PG = SPARK_SUBMIT.replace(
    "aws-java-sdk-bundle:1.12.262'",
    "aws-java-sdk-bundle:1.12.262,org.postgresql:postgresql:42.5.4'"
)

with DAG(
    dag_id="fraud_pipeline_dag",
    default_args=DEFAULT_ARGS,
    description="Main real-time fraud pipeline running every 15 minutes",
    schedule_interval="*/15 * * * *",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    tags=["fraud", "pipeline", "frequent"],
) as dag:

    # Task 1: Check Kafka lag
    check_kafka_lag = BashOperator(
        task_id="check_kafka_lag",
        bash_command=(
            "docker exec fraud-detection-pipeline-kafka-1 "
            "kafka-consumer-groups --bootstrap-server kafka:9092 "
            "--describe --all-groups 2>/dev/null | grep -E 'LAG|TOPIC' || "
            "echo 'Kafka lag check complete'"
        ),
    )

    # Task 2: Bronze Ingestion
    run_bronze_ingestion = BashOperator(
        task_id="run_bronze_ingestion",
        bash_command=SPARK_SUBMIT + "/workspace/bronze/ingest_to_bronze.py",
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.aws_access_key_id }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.aws_secret_access_key }}",
            "DB_PASSWORD": "{{ var.value.db_password }}",
        }
    )

    # Task 3: Silver Transformation
    run_silver_transform = BashOperator(
        task_id="run_silver_transform",
        bash_command=SPARK_SUBMIT + "/workspace/silver/transform_to_silver.py {{ ds }}",
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.aws_access_key_id }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.aws_secret_access_key }}",
            "DB_PASSWORD": "{{ var.value.db_password }}",
        }
    )

    # Task 4: Gold Inference
    run_gold_inference = BashOperator(
        task_id="run_gold_inference",
        bash_command=SPARK_SUBMIT_PG + "/workspace/gold/inference_to_gold.py",
        env={
            "AWS_ACCESS_KEY_ID": "{{ var.value.aws_access_key_id }}",
            "AWS_SECRET_ACCESS_KEY": "{{ var.value.aws_secret_access_key }}",
            "DB_PASSWORD": "{{ var.value.db_password }}",
        }
    )

    # Task 5: Completion Alert
    send_completion_alert = BashOperator(
        task_id="send_completion_alert",
        bash_command=(
            "curl -s -X POST $SLACK_WEBHOOK_URL "
            "-H 'Content-type: application/json' "
            "--data '{\"text\": \"Pipeline run complete for {{ ds }}\"}' || "
            "echo 'Pipeline complete {{ ds }}'"
        ),
        env={
            "SLACK_WEBHOOK_URL": "{{ var.value.slack_webhook_url }}",
        }
    )

    check_kafka_lag >> run_bronze_ingestion >> run_silver_transform >> run_gold_inference >> send_completion_alert