import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

def create_feature_store():
    # Initialize Spark Session with Delta Lake cluster support and wrapper
    builder = SparkSession.builder \
        .appName("GoldFeatureEngineering") \
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

    print("Spark Session initialized successfully with Delta & Cluster support!")

    silver_fact_path = "s3a://fraud-detection-lake-nouman/silver/fact_transactions/"
    silver_customer_path = "s3a://fraud-detection-lake-nouman/silver/dim_customer/"

    print(f"Reading Silver fact_transactions from: {silver_fact_path}")
    fact_df = spark.read.format("delta").load(silver_fact_path)

    print(f"Reading Silver dim_customer from: {silver_customer_path}")
    customer_df = spark.read.format("delta").load(silver_customer_path)

    print("Columns available in fact_df:", fact_df.columns)

    print("Joining fact_transactions with dim_customer...")

    feature_df = fact_df.join(customer_df, fact_df.customer_id == customer_df.customer_id, "inner") \
        .select(
            fact_df.transaction_id,
            fact_df.timestamp,
            fact_df.customer_id,
            fact_df.amount,
            fact_df.merchant_id,
            fact_df.oldbalanceOrg,
            fact_df.newbalanceOrig,
            fact_df.isFraud,
            fact_df.is_balance_fraud_signal,
            fact_df.is_data_inconsistency
        )

    # Cast flags to integer
    feature_df = feature_df \
        .withColumn("is_balance_fraud_signal", F.col("is_balance_fraud_signal").cast("integer")) \
        .withColumn("is_data_inconsistency", F.col("is_data_inconsistency").cast("integer"))

    print("\n--- Feature Engineering Preview ---")
    feature_df.printSchema()
    feature_df.show(5, truncate=False)

    gold_output_path = "s3a://fraud-detection-lake-nouman/gold/ml_features/"
    print(f"Writing processed ML features to Gold layer: {gold_output_path}")

    feature_df.write \
        .format("delta") \
        .mode("overwrite") \
        .save(gold_output_path)

    print("Feature Engineering complete and saved to Gold layer!")
    spark.stop()

if __name__ == "__main__":
    create_feature_store()