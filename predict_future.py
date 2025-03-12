import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Step 1: Create a SQLAlchemy engine for PostgreSQL
# Replace the connection details with your actual database credentials
engine = create_engine('postgresql+psycopg2://admin:new@localhost:5432/bioreactor')

# Step 2: Query the database to retrieve the sensor data for temperature
query = """
SELECT timestamp, temperature
FROM sensor_data
WHERE timestamp >= NOW() - INTERVAL '30 days'
ORDER BY timestamp ASC;
"""
# Fetch data into a pandas DataFrame
df = pd.read_sql(query, engine)

# Step 3: Prepare the dataframe for Prophet
# Prophet requires columns 'ds' (timestamp) and 'y' (target variable)
df_prophet = df.rename(columns={'timestamp': 'ds', 'temperature': 'y'})

# Step 4: Initialize and fit the Prophet model
model = Prophet()
model.fit(df_prophet)

# Step 5: Create a dataframe with future dates for prediction
# Forecast for 7 future days
future = model.make_future_dataframe(periods=7)

# Step 6: Predict the future values
forecast = model.predict(future)

# Step 7: Plot the forecast
# Save the plot to a file (for non-interactive environments)
model.plot(forecast)
plt.title('Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.savefig('temperature_forecast.png')  # Save the plot as an image file
# plt.show()  # Uncomment this if running in an interactive environment

# Step 8: Close the database connection (optional with SQLAlchemy)
engine.dispose()

print("Forecast completed and saved to 'temperature_forecast.png'.")
