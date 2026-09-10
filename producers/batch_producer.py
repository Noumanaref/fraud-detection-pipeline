import csv
import json
from kafka import KafkaProducer

KAFKA_TOPIC = "legacy_batch"
CSV_PATH = "data/PS_20174392719_1491204439457_log.csv"

# Optimized for high-throughput streaming
producer = KafkaProducer(
    bootstrap_servers=["localhost:9093"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    batch_size=65536,          # Increased batch size (64KB)
    linger_ms=20,              # Allow slightly longer accumulation for larger batches
    # buffer_memory=67108864     # 64MB buffer to prevent blocking on fast loops
)

print(f"Starting high-speed batch ingestion from {CSV_PATH} into '{KAFKA_TOPIC}'...")

count = 0
try:
    with open(CSV_PATH, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            payload = {
                "step": int(row["step"]),
                "transaction_type": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "isFraud": int(row["isFraud"]),
                "isFlaggedFraud": int(row["isFlaggedFraud"]),
            }

            # Asynchronous send (non-blocking)
            producer.send(KAFKA_TOPIC, value=payload)
            count += 1

            if count % 50000 == 0:
                print(f"Sent {count:,} records...")

except FileNotFoundError:
    print(f"Error: Could not find the CSV file at {CSV_PATH}.")
except KeyboardInterrupt:
    print("\nStopping batch ingestion early...")
finally:
    print("Flushing remaining messages to Kafka...")
    producer.flush()
    producer.close()
    print(f"Ingestion complete. Total records sent: {count:,}")