import json
import time
import random
from faker import Faker
from kafka import KafkaProducer

fake = Faker() ## Faker initialization

producer  = KafkaProducer (
    bootstrap_servers = ['localhost:9093'], ## this bootstrap_server will tell the producer where to find the broker.
    # api_version = (2,6,0),
    value_serializer = lambda v: json.dumps(v).encode('utf-8') ## automatically converts python dictionaries to JSON bytes before sending.
)

TOPIC_NAME = 'raw_transactions'
print(f"producer connected to kafka. Target Topic : {TOPIC_NAME}")


## lets generate fake transactions
def generate_transaction():
    """Generates a single fake transaction."""
    return {
        "transaction_id": fake.uuid4(),
        "timestamp": fake.iso8601(),
        "user_id": fake.uuid4(),
        "amount": round(random.uniform(5.0, 2000.0), 2),
        "merchant": fake.company(),
        "location": fake.city(),
        "transaction_type": random.choice(["purchase", "refund", "withdrawal", "deposit"]),
        "isFraud" : 0
    }

print("Starting live stream... Press Ctrl+C to stop.")

try:
    while True:
        # 1. Generate a fake transaction
        transaction = generate_transaction()
        
        # 2. Send it to Kafka
        producer.send(TOPIC_NAME, value=transaction)
        
        # 3. Print a short confirmation to the terminal
        print(f"Sent ID: {transaction['transaction_id'][:8]}... | Amount: ${transaction['amount']}")
        
        # 4. Sleep briefly to simulate ~10-100 events per second
        time.sleep(random.uniform(0.01, 0.1))
        
except KeyboardInterrupt:
    print("\nStopping producer gracefully...")
finally:
    # Ensure all messages are delivered before closing
    producer.flush()
    producer.close()