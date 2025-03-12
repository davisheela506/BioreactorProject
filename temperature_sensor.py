from kafka import KafkaProducer
import time
import random

# Create a Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Function to generate fake sensor data
def generate_sensor_data():
    temperature = round(random.uniform(20.0, 25.0), 1)  # Generate a random temperature
    pressure = round(random.uniform(1.0, 1.5), 2)  # Generate a random pressure
    humidity = random.randint(40, 60)  # Generate random humidity
    o2_level = round(random.uniform(19.0, 21.0), 1)  # Generate a random oxygen value

    # Create a sensor data string in the format "temperature=...,pressure=...,humidity=...,o2=..."
    sensor_data = f"temperature={temperature};pressure={pressure};humidity={humidity};o2={o2_level}"

    return sensor_data

# Send data every 5 seconds
while True:
    data = generate_sensor_data()
    # Send the data to Kafka topic 'bioreactor-sensor-data'
    producer.send('bioreactor-sensor-data', value=data.encode())
    print(f"Sent data: {data}")
    time.sleep(5)
 
