import os
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col, expr, to_timestamp, hour, dayofmonth, dayofweek, when, col
from pyspark.sql.functions import sha2, concat_ws, lit, current_timestamp, coalesce

# Configuration && partition prunning
# for now we will hard_code the partition prunning later airflow in M7 will pass this dynamically

YEAR = 2026
MONTH = 8
DAY = 15


BRONZE_RAW_TX_PATH = f"s3a://fraud-detection-lake-nouman-v2/bronze/raw_transactions/year={YEAR}/month={MONTH}/day={DAY}/"
BRONZE_LEGACY_PATH = f"s3a://fraud-detection-lake-nouman-v2/bronze/legacy_batch/year={YEAR}/month={MONTH}/day={DAY}/"


# initialize sparkSession with delta-lake
builder = SparkSession.builder \
    .appName("SilverTransformation") \
    .master("local[2]") \
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
    .withColumn("transaction_id", sha2(concat_ws("_", col("customer_id"), col("merchant_id"), col("amount"), col("step")), 256))  \
    .withColumn("timestamp", expr("timestamp('2026-07-01 00:00:00') + interval 1 hour * step")) \
    .drop("step")


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
    .withColumn("is_data_inconsistency", expr("isFlaggedFraud != isFraud")) 



# --- Verification ---
print("\n--- Cleaned Silver Data Preview ---")
print(f"Row count after cleaning: {silver_df.count()}")
silver_df.select("transaction_id", "amount", "is_balance_fraud_signal").show(5, truncate=False)



# step D : Lets build star_schema where we have transactions as fact_table
print("\n--- Building Star Schema Dimensions ---")


dim_time = silver_df.select("timestamp").distinct() \
    .withColumn("time_id", sha2(col("timestamp").cast("string"), 256)) \
    .withColumn("full_timestamp", col("timestamp")) \
    .withColumn("hour", hour("timestamp")) \
    .withColumn("day", dayofmonth("timestamp")) \
    .withColumn("month", expr("month(timestamp)")) \
    .withColumn("year", expr("year(timestamp)")) \
    .withColumn("day_of_week", col("timestamp").cast("string")) \
    .withColumn("is_weekend", when(dayofweek("timestamp").isin(1, 7), True).otherwise(False)) \
    .dropDuplicates(["time_id"])



# 2. dim_user (SCD Type 2 Layout): Add tracking attributes and default validity flags
dim_user = silver_df.select("customer_id").distinct() \
    .withColumn("user_id", col("customer_id")) \
    .withColumn("customer_name", lit("Unknown User")) \
    .withColumn("risk_tier", lit("Standard")) \
    .withColumn("registered_location", lit("Sialkot, PK")) \
    .withColumn("account_age_days", lit(365)) \
    .withColumn("valid_from", current_timestamp()) \
    .withColumn("valid_to", lit(None).cast("timestamp")) \
    .withColumn("is_current", lit(True)) \
    .drop("customer_id") \
    .dropDuplicates(["user_id"])


# 3. dim_merchant (SCD Type 1 Layout): Add descriptive attributes
dim_merchant = silver_df.select("merchant_id", "transaction_type").distinct() \
    .withColumn("merchant_id", sha2(col("merchant_id"), 256)) \
    .withColumn("merchant_name", lit("Retail Partner")) \
    .withColumn("merchant_category", col("transaction_type")) \
    .withColumn("terminal_location", lit("Online")) \
    .withColumn("channel", lit("Digital")) \
    .drop("transaction_type") \
    .dropDuplicates(["merchant_id"])


# 4. fact_fraud_inference: Align fact table columns with foreign key hashes

fact_fraud_inference = silver_df \
    .withColumn("user_id", col("customer_id")) \
    .withColumn("merchant_id", sha2(col("merchant_id"), 256)) \
    .withColumn("time_id", sha2(col("timestamp").cast("string"), 256)) \
    .withColumn("model_id", lit("placeholder_model_id")) \
    .select(
        "transaction_id", "user_id", "merchant_id", "time_id", "model_id",
        col("amount").alias("transaction_amount"),
        "oldbalanceOrg",
        "newbalanceOrig",
        "is_balance_fraud_signal",
        col("isFraud").alias("is_fraud"),
        lit(0.0).alias("inference_latency_ms"),
        col("timestamp").alias("inference_timestamp")
    )


print("\n--- Star Schema Row Counts ---")
print(f"fact_fraud_inference rows: {fact_fraud_inference.count()}")
print(f"dim_time rows: {dim_time.count()}")
print(f"dim_user rows: {dim_user.count()}")
print(f"dim_merchant rows: {dim_merchant.count()}")


print("\n--- Spark Execution Plan for fact_fraud_inference ---")
fact_fraud_inference.explain(mode="formatted")


# Data Quality Checks-
def run_quality_checks(df, table_name):
    print(f"\n--- Data Quality Checks: {table_name} ---")
    
    total_rows = df.count()
    
    # Check 1: No null transaction_ids
    null_ids = df.filter(col("transaction_id").isNull()).count()
    assert null_ids == 0, f"FAILED: {null_ids} null transaction_ids found"
    print(f"No null transaction_ids ({total_rows} rows)")
    
    # Check 2: Amount must be positive
    negative_amounts = df.filter(col("transaction_amount") <= 0).count()
    assert negative_amounts == 0, f"FAILED: {negative_amounts} non-positive amounts"
    print(f"All amounts positive")
    
    # Check 3: isFraud must be 0 or 1
    invalid_fraud = df.filter(~col("is_fraud").isin([0, 1])).count()
    assert invalid_fraud == 0, f"FAILED: {invalid_fraud} invalid isFraud values"
    print(f"isFraud values valid (0 or 1 only)")
    
    # Check 4: Row count sanity check
    assert total_rows > 1000, f"FAILED: Only {total_rows} rows — suspiciously low"
    assert total_rows < 10_000_000, f"FAILED: {total_rows} rows — suspiciously high"
    print(f"Row count within expected range: {total_rows}")
    
    print(f"All quality checks passed for {table_name}")

# Execute quality gates before writing to storage
run_quality_checks(fact_fraud_inference, "fact_fraud_inference")


# --- Step E: Write to Silver as Delta Lake ---

FACT_PATH = "s3a://fraud-detection-lake-nouman-v2/silver/fact_fraud_inference/"
TIME_PATH = "s3a://fraud-detection-lake-nouman-v2/silver/dim_time/"
USER_PATH = "s3a://fraud-detection-lake-nouman-v2/silver/dim_user/"
MERCHANT_PATH = "s3a://fraud-detection-lake-nouman-v2/silver/dim_merchant/"

# 1. Fact Table (Idempotent Append/Merge) 

print("Writing fact_fraud_inference...")

if DeltaTable.isDeltaTable(spark, FACT_PATH):
    delta_fact = DeltaTable.forPath(spark, FACT_PATH)
    delta_fact.alias("target").merge(
        fact_fraud_inference.alias("source"),
        "target.transaction_id = source.transaction_id"
    ).whenNotMatchedInsertAll().execute()
else:
    fact_fraud_inference.write.format("delta").mode("overwrite").save(FACT_PATH)




# 2. Dim Time (Merge to avoid duplicates)
print("Writing dim_time...")
if DeltaTable.isDeltaTable(spark, TIME_PATH):
    delta_time = DeltaTable.forPath(spark, TIME_PATH)
    delta_time.alias("target").merge(
        dim_time.alias("source"),
        "target.time_id = source.time_id"
    ).whenNotMatchedInsertAll().execute()
else:
    dim_time.write.format("delta").mode("overwrite").save(TIME_PATH)


# 3. Dim Merchant - SCD Type 1 (Upsert/Overwrite attributes)  - UPSERT
print("Writing dim_merchant (SCD Type 1)...")
if DeltaTable.isDeltaTable(spark, MERCHANT_PATH):
    delta_merchant = DeltaTable.forPath(spark, MERCHANT_PATH)
    delta_merchant.alias("target").merge(
        dim_merchant.alias("source"),
        "target.merchant_id = source.merchant_id"
    ).whenMatchedUpdate(set={
        "merchant_name": col("source.merchant_name"),
        "merchant_category": col("source.merchant_category"),
        "terminal_location": col("source.terminal_location"),
        "channel": col("source.channel")
    }).whenNotMatchedInsertAll().execute()
else:
    dim_merchant.write.format("delta").mode("overwrite").save(MERCHANT_PATH)



# 4. Dim User - SCD Type 2 (History Tracking)
print("Writing dim_user (SCD Type 2)...")
if not DeltaTable.isDeltaTable(spark, USER_PATH):
    dim_user.write.format("delta").mode("overwrite").save(USER_PATH)
else:
    # SCD Type 2 Merge Strategy: Expire old rows and insert updated rows
    delta_user = DeltaTable.forPath(spark, USER_PATH)
    
    dim_user.createOrReplaceTempView("incoming_users")
    
    spark.sql(f"""
        MERGE INTO delta.`{USER_PATH}` target
        USING incoming_users source
        ON target.user_id = source.user_id AND target.is_current = true
        WHEN MATCHED AND (target.risk_tier != source.risk_tier OR target.registered_location != source.registered_location) THEN
          UPDATE SET target.is_current = false, target.valid_to = current_timestamp()
    """)
    
    # Insert brand new records or newly expired record variants as current active rows
    delta_user.alias("target").merge(
        dim_user.alias("source"),
        "target.user_id = source.user_id AND target.is_current = true"
    ).whenNotMatchedInsertAll().execute()





print("--- Silver Transformation & Star Schema S3 Loading Complete! ---")