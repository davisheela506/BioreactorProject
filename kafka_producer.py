import random  
import time  
from kafka import KafkaProducer  

# Initialize Kafka producer  
producer = KafkaProducer(bootstrap_servers='localhost:9092')  

# Function to generate random sensor data  
def generate_sensor_data():  
    temperature = round(random.uniform(20, 25), 1)  # Temperature in °C  
    pressure = round(random.uniform(1.0, 1.5), 2)   # Pressure in bar  
    humidity = round(random.uniform(40, 60), 0)     # Humidity in %  
    o2_level = round(random.uniform(19, 21), 1)     # O₂ level in %  

    # Format the data as a string  
    data = f"temperature={temperature};pressure={pressure};humidity={humidity};o2={o2_level}"  
    return data  

# Infinite loop to continuously send sensor data  
while True:  
    sensor_data = generate_sensor_data()  
    producer.send('bioreactor_data', sensor_data.encode('utf-8'))  # Send data to Kafka  
    print(f"Sent: {sensor_data}")  # Print for debugging  

    time.sleep(2)  # Wait for 2 seconds before sending the next data  
