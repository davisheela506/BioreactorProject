import psycopg2
import pandas as pd

# Database connection parameters
DB_PARAMS = {
    "dbname": "bioreactor",
    "user": "admin",
    "password": "new",  # Use your actual password
    "host": "localhost",
    "port": "5432"
}

# Connect to PostgreSQL and fetch data
def fetch_sensor_data():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        query = """
        SELECT timestamp, temperature, pressure, humidity , o2_level
        FROM sensor_data 
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        ORDER BY timestamp ASC;
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print("Error fetching data:", e)
        return None

# Fetch data and print sample
data = fetch_sensor_data()
if data is not None:
    print(data.head())  # Show first few rows
