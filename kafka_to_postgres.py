import psycopg2
from kafka import KafkaConsumer

# Connect to PostgreSQL
try:
    conn = psycopg2.connect("dbname=bioreactor user=admin password=new host=localhost port=5432")
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit()

# Kafka Consumer Setup
try:
    consumer = KafkaConsumer(
        'bioreactor-sensor-data',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda x: x.decode('utf-8')
    )
    print("✅ Connected to Kafka topic: bioreactor-sensor-data")
except Exception as e:
    print(f"❌ Kafka connection failed: {e}")
    exit()

def detect_anomalies(temperature, pressure, humidity, o2):
    anomalies = []
    if temperature > 26 or temperature < 19:
        anomalies.append(f"🔥 Temperature anomaly detected: {temperature}°C")
    if pressure < 1.0 or pressure > 1.5:
        anomalies.append(f"⚠️ Pressure anomaly detected: {pressure} bar")
    if humidity < 40 or humidity > 60:
        anomalies.append(f"🌫️ Humidity anomaly detected: {humidity}%")
    if o2 < 18 or o2 > 22:
        anomalies.append(f"🫁 O₂ anomaly detected: {o2}%")
    return anomalies

print("🚀 Listening for messages...")
for message in consumer:
    try:
        data = message.value.strip()
        print(f"📩 Received: {data}")

        # Validate data format
        parts = data.split(";")
        if len(parts) != 4:
            print(f"❌ Corrupted data skipped: {data}")
            continue

        # Extract sensor values
        temperature = float(parts[0].split("=")[1])
        pressure = float(parts[1].split("=")[1])
        humidity = float(parts[2].split("=")[1])
        o2 = float(parts[3].split("=")[1])

        print(f"🌡️ Temp: {temperature}°C, ⏲️ Pressure: {pressure} bar, 💧 Humidity: {humidity}%, O₂: {o2}%")

        # Detect anomalies
        anomalies = detect_anomalies(temperature, pressure, humidity, o2)

        # Store anomalies in PostgreSQL
        for anomaly in anomalies:
            cursor.execute("INSERT INTO anomalies (timestamp, message) VALUES (NOW(), %s)", (anomaly,))
            conn.commit()
            print(f"✅ Anomaly saved: {anomaly}")

    except Exception as e:
        print(f"❌ Error processing message: {e}")

conn.close()
print("🔴 Connection closed")

