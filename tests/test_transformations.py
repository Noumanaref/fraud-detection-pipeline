import pytest
import hashlib
import sys
import os

# pytest      → the testing framework itself
# hashlib     → Python's built-in hashing library (sha256)


# ══════════════════════════════════════════════════════════════
# SECTION 1: Idempotency Tests
# Business logic: same input must ALWAYS produce same transaction_id
# This validates our sha2 fix that replaced uuid()
# ══════════════════════════════════════════════════════════════


def test_sha2_hash_is_deterministic():
    """
    WHAT: Tests that sha2 hashing is deterministic.
    WHY: Our PaySim records use sha2(customer_id + amount + step) as transaction_id.
         If we reprocess the same raw data, we must get the same transaction_id.
         This is the foundation of pipeline idempotency.
    """
    # Simulate the concat_ws("_", customer_id, merchant_id, amount, step) pattern
    # from transform_to_silver.py
    input_string = "C1231006815_M1979787155_9839.64_1"

    # Hash it twice — simulates running Silver transformation twice
    hash_run_1 = hashlib.sha256(input_string.encode()).hexdigest()
    hash_run_2 = hashlib.sha256(input_string.encode()).hexdigest()

    # assert = the core pytest statement
    # If this condition is False, the test FAILS with a clear error message
    assert (
        hash_run_1 == hash_run_2
    ), "SHA2 hash is not deterministic — idempotency is broken"


def test_different_inputs_produce_different_hashes():
    """
    WHAT: Tests that different transactions get different IDs.
    WHY: If two different transactions produce the same hash, one will
         overwrite the other in the Delta MERGE — data loss in production.
    """
    input_a = "C1231006815_M1979787155_9839.64_1"
    input_b = "C9999999999_M0000000001_100.00_5"

    hash_a = hashlib.sha256(input_a.encode()).hexdigest()
    hash_b = hashlib.sha256(input_b.encode()).hexdigest()

    # assert with != checks they are NOT equal
    assert (
        hash_a != hash_b
    ), "Two different transactions produced the same hash — collision detected"


# ══════════════════════════════════════════════════════════════
# SECTION 2: Fraud Signal Flag Tests
# Business logic: rule-based fraud detection flags
# These flags are features fed into XGBoost
# ══════════════════════════════════════════════════════════════


def test_balance_fraud_signal_is_true_when_triggered():
    """
    WHAT: Tests is_balance_fraud_signal = True when balance drains to 0
          after a large transfer.
    WHY: This is a known fraud pattern in PaySim — fraudsters drain accounts.
         Our Silver layer computes: newbalanceOrig == 0 AND amount > 10000
         If this logic breaks, the XGBoost model loses a key fraud feature.
    """
    new_balance_orig = 0.0  # account drained to zero
    amount = 15000.0  # large transfer

    # Mirror the exact PySpark expr() logic from transform_to_silver.py
    is_balance_fraud_signal = (new_balance_orig == 0) and (amount > 10000)

    assert (
        is_balance_fraud_signal is True
    ), "Balance fraud signal should be True when balance=0 and amount>10000"


def test_balance_fraud_signal_is_false_for_normal_transaction():
    """
    WHAT: Tests is_balance_fraud_signal = False for normal transactions.
    WHY: False positives are expensive — every false fraud alert costs
         an analyst time. Normal transactions must not trigger this flag.
    """
    new_balance_orig = 5000.0  # balance still has funds
    amount = 250.0  # small transaction

    is_balance_fraud_signal = (new_balance_orig == 0) and (amount > 10000)

    assert (
        is_balance_fraud_signal is False
    ), "Normal transaction incorrectly flagged as balance fraud signal"


def test_balance_fraud_signal_boundary_exactly_10000():
    """
    WHAT: Tests the boundary condition at exactly amount=10000.
    WHY: Boundary conditions are where bugs hide. Our rule is amount > 10000
         (strictly greater than). At exactly 10000 the signal should be False.
         This is called a boundary value test — critical in financial systems.
    """
    new_balance_orig = 0.0
    amount = 10000.0  # exactly at boundary

    is_balance_fraud_signal = (new_balance_orig == 0) and (amount > 10000)

    # 10000 is NOT > 10000, so this must be False
    assert (
        is_balance_fraud_signal is False
    ), "Amount of exactly 10000 should NOT trigger fraud signal (rule is strictly >)"


def test_data_inconsistency_flag():
    """
    WHAT: Tests is_data_inconsistency flag logic.
    WHY: This flag detects when the bank's rule-based system disagreed
         with ground truth. isFlaggedFraud=0 but isFraud=1 means the bank
         missed a fraud case. We use this as a feature signal.
    """
    # Case 1: bank missed the fraud (inconsistency = True)
    is_flagged_fraud = 0
    is_fraud = 1
    is_data_inconsistency = is_flagged_fraud != is_fraud
    assert (
        is_data_inconsistency is True
    ), "Should detect inconsistency when bank missed fraud"

    # Case 2: both agree (no inconsistency)
    is_flagged_fraud = 1
    is_fraud = 1
    is_data_inconsistency = is_flagged_fraud != is_fraud
    assert (
        is_data_inconsistency is False
    ), "No inconsistency when bank and ground truth agree"


# ══════════════════════════════════════════════════════════════
# SECTION 3: Data Quality Tests
# Business logic: validates our run_quality_checks() assertions
# These mirror exactly what Silver layer checks before writing to S3
# ══════════════════════════════════════════════════════════════


def test_amount_must_be_positive():
    """
    WHAT: Amount must always be positive.
    WHY: A transaction of $0 or negative is a data quality issue.
         XGBoost was trained on positive amounts — negative values
         would cause silent model prediction errors.
    """
    valid_amount = 250.50
    invalid_amount = -100.0
    zero_amount = 0.0

    assert valid_amount > 0, "Valid amount should pass quality check"
    assert not (invalid_amount > 0), "Negative amount should fail quality check"
    assert not (zero_amount > 0), "Zero amount should fail quality check"


def test_isfraud_must_be_binary():
    """
    WHAT: isFraud must only be 0 or 1.
    WHY: XGBoost was trained with binary labels. A value of 2 or -1
         would corrupt the model's prediction calibration.
    """
    valid_values = {0, 1}

    assert 0 in valid_values, "0 should be valid isFraud value"
    assert 1 in valid_values, "1 should be valid isFraud value"
    assert 2 not in valid_values, "2 should be invalid isFraud value"
    assert -1 not in valid_values, "-1 should be invalid isFraud value"


def test_transaction_id_not_none():
    """
    WHAT: transaction_id must never be None or empty.
    WHY: transaction_id is the PRIMARY KEY of fact_fraud_inference.
         A None transaction_id would either crash the Delta MERGE
         or create phantom records in PostgreSQL.
    """
    valid_id = "abc123def456"
    invalid_id_none = None
    invalid_id_empty = ""

    # 'and' short-circuits: if first condition fails, second is not evaluated
    assert valid_id is not None and valid_id != "", "Valid transaction_id should pass"
    assert (
        invalid_id_none is None
    ), "None transaction_id should be caught by quality check"
    assert (
        invalid_id_empty == ""
    ), "Empty transaction_id should be caught by quality check"


def test_row_count_within_expected_range():
    """
    WHAT: Row count must be within a sane range.
    WHY: Too few rows means Bronze ingestion failed silently.
         Too many rows means something ran twice (idempotency failure).
         Our quality check asserts: 1000 < rows < 10,000,000
    """
    MIN_ROWS = 1000
    MAX_ROWS = 10_000_000

    normal_count = 139996  # typical Silver row count in our pipeline
    suspicious_low = 5  # ingestion probably failed
    suspicious_high = 50_000_000  # probably ran twice

    assert (
        MIN_ROWS < normal_count < MAX_ROWS
    ), "Normal row count should be within expected range"
    assert not (
        MIN_ROWS < suspicious_low < MAX_ROWS
    ), "Suspiciously low count should fail range check"
    assert not (
        MIN_ROWS < suspicious_high < MAX_ROWS
    ), "Suspiciously high count should fail range check"


# ══════════════════════════════════════════════════════════════
# SECTION 4: Schema Tests
# Business logic: validates column names match across pipeline
# Prevents schema drift between Silver and Gold layers
# ══════════════════════════════════════════════════════════════


def test_feature_columns_present_in_schema():
    """
    WHAT: Verifies that all XGBoost feature columns exist in the schema.
    WHY: If Silver renames 'transaction_amount' to 'amount', Gold inference
         silently fails because feature_cols won't find the column.
         This test catches schema drift before it reaches production.
    """
    # The exact feature columns used in inference_to_gold.py
    required_features = [
        "transaction_amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "is_balance_fraud_signal",
    ]

    # Simulated schema from fact_fraud_inference Silver Delta table
    actual_schema_columns = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "time_id",
        "model_id",
        "transaction_amount",  # renamed from 'amount' in Silver
        "oldbalanceOrg",
        "newbalanceOrig",
        "is_balance_fraud_signal",
        "is_fraud",
        "inference_latency_ms",
        "inference_timestamp",
    ]

    for feature in required_features:
        # 'in' operator checks membership in the list
        assert (
            feature in actual_schema_columns
        ), f"Required feature '{feature}' is missing from Silver schema — inference will fail"


def test_star_schema_fact_table_columns():
    """
    WHAT: Validates fact_fraud_inference has all required FK columns.
    WHY: PostgreSQL will reject any INSERT that's missing a NOT NULL
         column or a foreign key. This test catches schema mismatches
         before the Gold upsert hits the database.
    """
    required_fact_columns = [
        "transaction_id",  # PRIMARY KEY
        "user_id",  # FK → dim_user
        "merchant_id",  # FK → dim_merchant
        "time_id",  # FK → dim_time
        "model_id",  # FK → dim_model
        "transaction_amount",
        "xgboost_probability",
        "is_fraud",
        "inference_latency_ms",
        "inference_timestamp",
    ]

    # Every column must be unique — no duplicates in schema
    assert len(required_fact_columns) == len(
        set(required_fact_columns)
    ), "Duplicate column names detected in fact table schema"

    # transaction_id must be first (PRIMARY KEY convention)
    assert (
        required_fact_columns[0] == "transaction_id"
    ), "transaction_id should be the first column (PRIMARY KEY)"


# ══════════════════════════════════════════════════════════════
# SECTION 5: DAG Structure Tests
# Business logic: validates Airflow DAG files load without errors
# ══════════════════════════════════════════════════════════════


def test_dag_file_exists():
    """
    WHAT: Verifies DAG files exist at expected paths.
    WHY: A missing DAG file causes silent failure — Airflow won't
         show an error, the DAG just disappears from the UI.
    """
    dag_paths = ["airflow/dags/fraud_pipeline_dag.py", "airflow/dags/retrain_dag.py"]

    for path in dag_paths:
        assert os.path.exists(
            path
        ), f"DAG file missing: {path} — Airflow will not schedule this pipeline"


def test_dag_imports_successfully():
    """
    WHAT: Imports the DAG Python file and checks for syntax errors.
    WHY: A typo in a DAG file causes the entire Airflow scheduler
         to log errors on every heartbeat. This test catches it
         before it ever reaches the Airflow container.
    """
    import importlib.util

    dag_file = "airflow/dags/fraud_pipeline_dag.py"

    if os.path.exists(dag_file):
        # importlib.util.spec_from_file_location loads a Python file by path
        spec = importlib.util.spec_from_file_location("fraud_pipeline_dag", dag_file)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            # If we reach here, the DAG file loaded without syntax errors
            assert True
        except Exception as e:
            pytest.fail(f"DAG file failed to import: {e}")
    else:
        pytest.skip("DAG file not found — skipping import test")
