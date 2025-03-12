from kafka import KafkaConsumer
import csv

# Create a Kafka consumer
consumer = KafkaConsumer(
    'bioreactor-sensor-data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True
)

# Open CSV file for writing
with open('bioreactor_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Temperature', 'Pressure', 'Humidity', 'Oxygen'])  # Add 'Oxygen' header

    for message in consumer:
        # Decode message from bytes to string
        data = message.value.decode('utf-8')
        
        # Extract values (including o2)
        values = data.replace("temperature=", "").replace("pressure=", "").replace("humidity=", "").replace("o2=", "").split(";")
        
        # Write to CSV
        writer.writerow(values)
        print(f"Saved: {values}")  # Print to confirm
