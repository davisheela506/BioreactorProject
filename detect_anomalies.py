import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

# Step 1: Create a SQLAlchemy engine for PostgreSQL
engine = create_engine('postgresql+psycopg2://admin:new@localhost:5432/bioreactor')

# Step 2: Query the database to retrieve recent sensor data
query = """
SELECT timestamp, temperature, pressure, humidity, o2_level
FROM sensor_data
WHERE timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp ASC;
"""
df = pd.read_sql(query, engine)

# Step 3: Ensure the column names match
print("Columns in DataFrame:", df.columns)  # Debugging

# Step 4: Select features for anomaly detection
features = df[['temperature', 'pressure', 'humidity', 'o2_level']]

# Step 5: Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination rate as needed
df['anomaly'] = model.fit_predict(features)

# Step 6: Extract anomalies
anomalies = df[df['anomaly'] == -1].copy()  # Make a copy to avoid SettingWithCopyWarning
anomalies.loc[:, 'message'] = "Anomalous sensor reading detected"

# Step 7: Insert anomalies into PostgreSQL
if not anomalies.empty:
    anomalies.to_sql('anomalies', engine, if_exists='append', index=False)
    print(f"Detected {len(anomalies)} anomalies and saved to database.")
else:
    print("No anomalies detected.")

# Step 8: Close database connection
engine.dispose()

