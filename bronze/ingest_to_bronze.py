from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, col

# --- 1. Configuration ---
KAFKA_BROKER = "kafka:9092"  # Internal Docker network — Spark runs inside Docker
KAFKA_TOPICS = "raw_transactions,legacy_batch"

BRONZE_BUCKET = "s3a://fraud-detection-lake-nouman"
BRONZE_PATH = f"{BRONZE_BUCKET}/bronze"
CHECKPOINT_PATH = f"{BRONZE_BUCKET}/checkpoints/bronze"

# --- 2. Initialize Spark Session ---
# hadoop-aws and aws-java-sdk-bundle allow Spark to talk to S3
spark = SparkSession.builder \
    .appName("BronzeIngestion") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session created successfully!")

# --- 3. Read from Kafka ---
# startingOffsets=earliest means read all messages from the beginning
# Kafka 'value' arrives as binary bytes — we cast to string in step 4
print(f"Connecting to Kafka at {KAFKA_BROKER}, topics: {KAFKA_TOPICS}")

raw_kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", KAFKA_TOPICS) \
    .option("startingOffsets", "earliest") \
    .load()

# --- 4. Extract Fields and Add Partition Columns ---
# Cast binary value to string (raw JSON payload)
# Extract year/month/day from Kafka timestamp for S3 date partitioning
bronze_df = raw_kafka_df.select(
    col("topic"),
    col("value").cast("string").alias("raw_payload"),
    col("timestamp")
).withColumn("year", year(col("timestamp"))) \
 .withColumn("month", month(col("timestamp"))) \
 .withColumn("day", dayofmonth(col("timestamp")))

# --- 5. Split into Two DataFrames by Topic ---
# Each topic gets its own S3 path and checkpoint to stay cleanly separated
raw_transactions_df = bronze_df.filter(col("topic") == "raw_transactions")
legacy_batch_df = bronze_df.filter(col("topic") == "legacy_batch")

# --- 6. Write raw_transactions Stream to Bronze ---
print(f"Starting stream 1: raw_transactions -> {BRONZE_PATH}/raw_transactions")

query1 = raw_transactions_df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", f"{BRONZE_PATH}/raw_transactions") \
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/raw_transactions") \
    .partitionBy("year", "month", "day") \
    .start()

# --- 7. Write legacy_batch Stream to Bronze ---
print(f"Starting stream 2: legacy_batch -> {BRONZE_PATH}/legacy_batch")

query2 = legacy_batch_df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", f"{BRONZE_PATH}/legacy_batch") \
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/legacy_batch") \
    .partitionBy("year", "month", "day") \
    .start()

# --- 8. Keep Both Streams Running ---
# awaitAnyTermination stops if either stream fails
spark.streams.awaitAnyTermination()