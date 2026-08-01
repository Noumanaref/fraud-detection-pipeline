import os
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col, expr, to_timestamp, hour, dayofmonth, dayofweek, when, col


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


# Step C : DataCleaning and feature flags

print("\n--- Step C: Cleaning Data & Adding Fraud Flags ---")

# drop duplicate transaction_ids
cleaned_df = unified_df.dropDuplicates(["transaction_id"])

# Handle missing values
# Fill missing isFlaggedFraud with 0, then drop rows missing critical keys

cleaned_df = cleaned_df.fillna({"isFlaggedFraud": 0})
cleaned_df = cleaned_df.dropna(subset = ["transaction_id","isFraud"])

## Add rule based fraud flags
# Use expr() to evaluate boolean logic conditions natively

silver_df = cleaned_df \
    .withColumn("is_balance_fraud_signal", expr("newbalanceOrig == 0 AND amount > 10000")) \
    ##.withColumn("is_data_inconsistency", expr("isFlaggedFraud != isFraud")) - caused data leakage



# --- Verification ---
print("\n--- Cleaned Silver Data Preview ---")
print(f"Row count after cleaning: {silver_df.count()}")
silver_df.select("transaction_id", "amount", "is_balance_fraud_signal").show(5, truncate=False)



# step D : Lets build star_schema where we have transactions as fact_table
print("\n--- Step D: Building Star Schema Dimensions ---")


# 1. dim_time: Extract time-based features from timestamp

dim_time = silver_df.select("timestamp").distinct() \
    .withColumn("hour", hour("timestamp")) \
    .withColumn("day", dayofmonth("timestamp")) \
    .withColumn("day_of_week",dayofweek("timestamp")) \
    .withColumn("is_weekend", when(col("day_of_week").isin( 1, 7), True).otherwise(False))


# 2. dim_customer: Extract unique customer information
dim_customer = silver_df.select("customer_id").dropDuplicates(["customer_id"])

# 3. dim_merchant: Extract unique merchant information
dim_merchant = silver_df.select("merchant_id", "transaction_type").dropDuplicates(["merchant_id"])

# 4. fact_transactions: The core event table (one row per transaction)
fact_transactions = silver_df.select(
    "transaction_id", "timestamp", "customer_id", "merchant_id",
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "is_balance_fraud_signal",
    "isFlaggedFraud", "isFraud"
)


print("\n--- Star Schema Row Counts ---")
print(f"fact_transactions rows: {fact_transactions.count()}")
print(f"dim_time rows: {dim_time.count()}")
print(f"dim_customer rows: {dim_customer.count()}")
print(f"dim_merchant rows: {dim_merchant.count()}")

    


# --- Step E: Write to Silver as Delta Lake ---

print("Writing fact_transactions...")
fact_transactions.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3a://fraud-detection-lake-nouman/silver/fact_transactions/")

print("Writing dim_time...")
dim_time.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3a://fraud-detection-lake-nouman/silver/dim_time/")

print("Writing dim_customer...")
dim_customer.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3a://fraud-detection-lake-nouman/silver/dim_customer/")

print("Writing dim_merchant...")
dim_merchant.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3a://fraud-detection-lake-nouman/silver/dim_merchant/")

print("--- Silver Transformation Complete! ---")
