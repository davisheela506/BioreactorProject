from kafka import KafkaConsumer
import psycopg2

# Kafka consumer to listen to 'bioreactor_data' topic
consumer = KafkaConsumer(
    'bioreactor_data', 
    bootstrap_servers='localhost:9092', 
    auto_offset_reset='earliest', 
    enable_auto_commit=True
)

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="bioreactor",
    user="admin",
    password="bioreactor123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data (
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        temperature FLOAT,
        pressure FLOAT,
        humidity FLOAT,  -- Changed INT to FLOAT
        o2_level FLOAT
    );
""")
conn.commit()

# Consume and store data
for message in consumer:
    data = message.value.decode('utf-8')
    
    # Extract values from the message
    temp, pressure, humidity, o2_level = data.split(';')
    temp_value = float(temp.split('=')[1])
    pressure_value = float(pressure.split('=')[1])
    humidity_value = float(humidity.split('=')[1])  # Changed int to float
    o2_value = float(o2_level.split('=')[1])

    # Insert data into PostgreSQL table
    cursor.execute("""
        INSERT INTO sensor_data (temperature, pressure, humidity, o2_level) 
        VALUES (%s, %s, %s, %s);
    """, (temp_value, pressure_value, humidity_value, o2_value))
    conn.commit()

    print(f"Stored data: {data}")

# Close the connection
cursor.close()
conn.close()

