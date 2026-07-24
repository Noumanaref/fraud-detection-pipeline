import os
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


# Configuration && partition prunning
# for now we will hard_code the partition prunning later airflow in M7 will pass this dynamically

YEAR = 2026
MONTH = 7
DAY = 20


BRONZE_RAW_TX_PATH = f"s3a://fraud-detection-lake-nouman/bronze/raw_transactions/year={YEAR}/month={MONTH}/day={DAY}/"
BRONZE_LEGACY_PATH = f"s3a://fraud-detection-lake-nouman/bronze/legacy_batch/year={YEAR}/month={MONTH}/day={DAY}/"


# initialize sparkSession with delta-lake
builder = SparkSession.builder \
    .appName("SilverTransformation") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages",
        "io.delta:delta-spark_2.12:3.1.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"]) \
    .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"]) \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
spark = configure_spark_with_delta_pip(builder).getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session created successfully with Delta Lake support!")


# Read from Bronze Incremental

print(f"Reading raw_transactions from: {BRONZE_RAW_TX_PATH}")
raw_tx_df = spark.read.parquet(BRONZE_RAW_TX_PATH)

print(f"Reading legacy_batch from: {BRONZE_LEGACY_PATH}")
legacy_df = spark.read.parquet(BRONZE_LEGACY_PATH)

# --- Verification ---
print("\n--- Raw Transactions ---")
raw_tx_df.printSchema()
print(f"Row count: {raw_tx_df.count()}")

print("\n--- Legacy Batch ---")
legacy_df.printSchema()
print(f"Row count: {legacy_df.count()}")




