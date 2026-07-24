import os
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col, expr, to_timestamp


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




# Step B : lets parse JSON and standardize schema
print("\n--- Step B: Unpacking JSON & Standardizing Schemas ---")

# Define schemas of both sources for from_json
mockingbird_schema = StructType(
    [StructField("transaction_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("merchant", StringType(), True),
    StructField("location", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("isFraud", IntegerType(), True)
    ])


paysim_schema = StructType([
    StructField("step", IntegerType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True)
    ])

# Parse and unpack JSON (raw_payload file of string type)
parsed_tx_df = raw_tx_df.withColumn("data", from_json(col("raw_payload") , mockingbird_schema)).select("data.*")
parsed_legacy_df = legacy_df.withColumn("data", from_json(col("raw_payload") , paysim_schema)).select("data.*")


# lets standardize schemas one by one
std_tx_df = parsed_tx_df\
.withColumnRenamed("user_id","customer_id")\
.withColumnRenamed("merchant","merchant_id")\
.withColumn("timestamp", to_timestamp(col("timestamp")))\
.drop("location")



std_legacy_df = parsed_legacy_df \
    .withColumnRenamed("nameOrig", "customer_id") \
    .withColumnRenamed("nameDest", "merchant_id") \
    .withColumn("transaction_id", expr("uuid()")) \
    .withColumn("timestamp", expr("timestamp('2026-07-01 00:00:00') + interval 1 hour * step")) \
    .drop("step")

# UUID = Universally Unique Identifier.

# lets unify the schema.
unified_df = std_tx_df.unionByName(std_legacy_df, allowMissingColumns=True)

print("\n--- Unified Schema ---")
unified_df.printSchema()