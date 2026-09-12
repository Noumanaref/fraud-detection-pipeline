## 1. Executive Summary
This project bridges the gap between raw data engineering and production machine learning (MLOps). By ingesting over **3.5 million transaction records**, the system cleans data through a structured lakehouse framework, tracks model versions using MLflow, versions data files using DVC, and pushes real-time alerts to operations teams.

---

## 2. Data Pipeline Architecture (Medallion Pattern)
The project organizes data into three distinct layers on AWS S3 to separate raw storage from business reporting:

* **Bronze Layer (Raw):** Stores incoming streaming and batch JSON/CSV data as append-only Parquet files partitioned by date. No cleaning happens here, ensuring a permanent audit trail.

* **Silver Layer (Cleaned & Unified):** Parses JSON payloads, unifies schemas from different sources, removes duplicates, handles data quality checks, and organizes data into a **Star Schema** (Fact and Dimension tables) using Delta Lake.

* **Gold Layer (Business-Ready):** Houses the ML feature store (`s3a://.../gold/ml_features/`) used directly for training models, alongside scored tables sent to PostgreSQL for dashboard tracking.

---

## 3. Core Architectural Decisions

### ELT Over ETL
* **Decision:** Extract, Load, and Transform (ELT) instead of traditional ETL.
* **Why:** Cloud storage (AWS S3) is inexpensive. Keeping raw data in the Bronze layer means if a transformation bug occurs, developers can fix the code and replay data from the source without losing historical records.

### Kappa Architecture
* **Decision:** Use Apache Kafka as a single unified pipeline for both real-time data and historical batch files.
* **Why:** Traditional systems separate real-time and batch into two codebases, which causes maintenance issues. Treating historical CSV files as a stream through Kafka unifies the processing logic.

### Kimball Dimensional Modeling
* **Decision:** Build a star schema using Ralph Kimball's bottom-up method.
* **Why:** Organizes analytical data around business processes (Fraud Inference) with a clear grain (one row equals one scored transaction) and distinct dimension tables (`dim_user`, `dim_merchant`, `dim_time`, `dim_model`), making queries fast and simple.

---

## 4. Optimization and Performance Engineering
Processing 3.5 million+ records locally or on limited cloud nodes presents severe memory challenges[cite: 2]. The following optimizations solved those bottlenecks:

* **Resource Expansion:** Moved Spark from a restricted 2-core setting (`local[2]`) to utilize all available hardware threads (`local[*]`) and allocated **6GB of driver memory** (`spark.driver.memory = "6g"`).
* **Elimination of Redundant Scans:** Fixed an issue where multiple `.count()` calls forced Spark to rescan the dataset from scratch. By introducing `.cache()`, the clean dataset is stored in RAM once, slashing execution time.
* **Shuffle Partition Tuning:** Changed Spark's default 200 shuffle partitions down to `8` to match physical CPU cores, stopping unnecessary inter-process communication delays.
* **Pandas Conversion Safety:** Dropped heavy, high-cardinality text columns (like transaction IDs and customer names) using `.select()` right before converting Spark tables to Pandas for XGBoost, preventing driver Out-Of-Memory (OOM) crashes.

---

## 5. Slowly Changing Dimensions (SCD)
To track how customer and merchant attributes change over time (such as a user changing risk tiers or locations), the pipeline implements:
* **SCD Type 1 (dim_merchant):** Overwrites old attributes when updates occur, as historical merchant category changes are less critical for immediate fraud checks.
* **SCD Type 2 (dim_user):** Inserts a new row and updates timestamps (`valid_from`, `valid_to`, `is_current`) when customer risk details change, preserving historical context for accurate past fraud audits.

---
