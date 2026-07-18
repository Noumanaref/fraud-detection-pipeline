import csv
import json
import time
from kafka import KafkaProducer

KAFKA_TOPIC = 'legacy_batch'
CSV_PATH = 'data/PS_20174392719_1491204439457_log.csv'

producer = KafkaProducer(
    bootstrap_servers=['localhost:9093'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    batch_size=16384,  # batch messages together for efficiency
    linger_ms=10       # wait 10ms to fill batch before sending
)

print(f"Starting batch ingestion from {CSV_PATH} into '{KAFKA_TOPIC}'...")

count = 0
try:
    with open(CSV_PATH, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            payload = {
                "step": int(row["step"]),
                "transaction_type": row["type"],  # renamed to match real-time schema
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "isFraud": int(row["isFraud"]),
                "isFlaggedFraud": int(row["isFlaggedFraud"])
            }
            
            producer.send(KAFKA_TOPIC, value=payload)
            count += 1
            
            if count % 10000 == 0:
                print(f"Sent {count} records...")
                producer.flush()
            
            time.sleep(0.001)  # ~1000 rows/second

except FileNotFoundError:
    print(f"Error: Could not find the CSV file at {CSV_PATH}.")
except KeyboardInterrupt:
    print("\nStopping batch ingestion early...")
finally:
    producer.flush()
    producer.close()
    print(f"Ingestion complete. Total records sent: {count}")