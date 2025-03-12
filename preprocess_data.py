import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import psycopg2

# Database connection parameters
DB_PARAMS = {
    "dbname": "bioreactor",
    "user": "admin",
    "password": "bioreactor123",  # Use your actual password
    "host": "localhost",
    "port": "5432"
}

# Fetch data from PostgreSQL
def fetch_sensor_data():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        query = """
        SELECT timestamp, temperature, pressure, humidity, o2_level 
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

# Preprocess data
def preprocess_data(df):
    df = df.dropna()  # Remove missing values
    df.set_index("timestamp", inplace=True)
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    return df, df_scaled

# Fetch and preprocess data
data = fetch_sensor_data()
if data is not None:
    raw_data, processed_data = preprocess_data(data)
    print("Preprocessed Data Sample:\n", raw_data.head())

