import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, col

# --- 1. Configuration ---
KAFKA_BROKER = "localhost:9093"
KAFKA_TOPICS = "raw_transactions,legacy_batch"

BRONZE_BUCKET = "s3a://fraud-detection-lake-nouman-v2"
BRONZE_PATH = f"{BRONZE_BUCKET}/bronze"
CHECKPOINT_PATH = f"{BRONZE_BUCKET}/checkpoints/bronze"

# --- 2. Initialize Spark Session ---
spark = SparkSession.builder \
    .appName("BronzeIngestion") \
    .master("local[2]") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"]) \
    .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"]) \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session created successfully!")

# --- 3. Read from Kafka ---
print(f"Connecting to Kafka at {KAFKA_BROKER}, topics: {KAFKA_TOPICS}")

raw_kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", KAFKA_TOPICS) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .option("kafka.security.protocol", "PLAINTEXT") \
    .load()

# --- 4. Extract Fields and Add Partition Columns ---
bronze_df = raw_kafka_df.select(
    col("topic"),
    col("value").cast("string").alias("raw_payload"),
    col("timestamp")
).withColumn("year", year(col("timestamp"))) \
 .withColumn("month", month(col("timestamp"))) \
 .withColumn("day", dayofmonth(col("timestamp")))

# --- 5. Split into Two DataFrames by Topic ---
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
    .trigger(processingTime="30 seconds") \
    .start()

# --- 7. Write legacy_batch Stream to Bronze ---
print(f"Starting stream 2: legacy_batch -> {BRONZE_PATH}/legacy_batch")

query2 = legacy_batch_df.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", f"{BRONZE_PATH}/legacy_batch") \
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/legacy_batch") \
    .partitionBy("year", "month", "day") \
    .trigger(processingTime="30 seconds") \
    .start()

# --- 8. Keep Both Streams Running ---
spark.streams.awaitAnyTermination()