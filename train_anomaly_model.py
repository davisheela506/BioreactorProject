import os
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Connect to PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        dbname="bioreactor", user="admin", password="new", host="localhost", port="5432"
    )

# Fetch sensor data from PostgreSQL
def fetch_sensor_data():
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT timestamp, temperature, pressure, humidity, o2_level FROM sensor_data ORDER BY timestamp")
            rows = cursor.fetchall()
    
    df = pd.DataFrame(rows, columns=["timestamp", "temperature", "pressure", "humidity", "o2_level"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop NaN values (important for training)
    df.dropna(inplace=True)
    return df

# Anomaly Detection using Isolation Forest
def detect_anomalies(df):
    logging.info("Running anomaly detection...")

    # Train Isolation Forest model
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[["temperature", "pressure", "humidity", "o2_level"]])

    # Filter anomalies
    anomalies_df = df[df["anomaly"] == -1]
    anomaly_count = len(anomalies_df)
    logging.info(f"Detected {anomaly_count} anomalies.")

    # Save anomalies to database
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            for _, row in anomalies_df.iterrows():
                cursor.execute(
                    "INSERT INTO anomalies (timestamp, temperature, pressure, humidity, o2_level, anomaly) VALUES (%s, %s, %s, %s, %s, %s)",
                    (row["timestamp"], row["temperature"], row["pressure"], row["humidity"], row["o2_level"], 1)
                )
        conn.commit()

    logging.info("Anomalies saved to the database.")

    # Generate and save recommendations
    if anomaly_count > 0:
        recommendations = generate_recommendations(anomalies_df)
        save_recommendations_to_db(recommendations)
        logging.info("✅ Operator recommendations generated and saved to the database.")

# Recommendation Generation Logic
def generate_recommendations(anomalies_df):
    recommendations = []

    for _, row in anomalies_df.iterrows():
        timestamp = row["timestamp"]
        temp, pressure, humidity, o2 = row["temperature"], row["pressure"], row["humidity"], row["o2_level"]
        advice = []

        # Temperature Check
        if temp > 40:
            advice.append("🔥 High temperature detected! Consider increasing cooling or reducing heat source.")
        elif temp < 10:
            advice.append("❄️ Low temperature detected! Check heating elements.")

        # Pressure Check
        if pressure > 1.5:
            advice.append("⚠️ High pressure detected! Check for valve malfunctions.")
        elif pressure < 0.5:
            advice.append("🛠️ Low pressure detected! Verify pump operation.")

        # Humidity Check
        if humidity > 80:
            advice.append("🌫️ High humidity detected! Check dehumidifiers.")
        elif humidity < 30:
            advice.append("💧 Low humidity detected! Inspect humidifiers.")

        # Oxygen Level Check
        if o2 < 10:
            advice.append("🫁 Low oxygen detected! Increase aeration system efficiency.")

        # Store recommendation
        if advice:
            recommendations.append((timestamp, temp, pressure, humidity, o2, "; ".join(advice)))

    return recommendations

# Save recommendations to PostgreSQL database
def save_recommendations_to_db(recommendations):
    with connect_to_db() as conn:
        with conn.cursor() as cursor:
            for rec in recommendations:
                cursor.execute(
                    "INSERT INTO recommendations (timestamp, temperature, pressure, humidity, o2_level, recommendation) VALUES (%s, %s, %s, %s, %s, %s)",
                    rec
                )
        conn.commit()
    logging.info("✅ Recommendations saved to database.")

# Preprocess data for LSTM
def preprocess_data(df):
    df.set_index('timestamp', inplace=True)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['temperature', 'pressure', 'humidity', 'o2_level']])

    return scaled_data, scaler

# Create sequences for LSTM
def create_sequences(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_lstm_model(X_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50, return_sequences=True),
        BatchNormalization(),  # Helps stabilize training
        LSTM(units=50, return_sequences=False),
        Dense(units=4)  # Predicting 4 variables (temperature, pressure, humidity, O2)
    ])
    
    # Configure optimizer with gradient clipping
    optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)  # Reduced learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Train LSTM Model
def train_lstm_model(X_train, y_train, epochs=10, batch_size=32):
    model = build_lstm_model(X_train)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

# Evaluate the LSTM Model
def evaluate_lstm_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predicted_data = scaler.inverse_transform(predictions)
    actual_data = scaler.inverse_transform(y_test)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(actual_data[:, 0], label='Actual Temperature')
    plt.plot(predicted_data[:, 0], label='Predicted Temperature', linestyle='dashed')
    plt.legend()
    plt.title("Temperature Forecasting")
    plt.savefig('forecast_plot.png')  # Save plot to file
    plt.show()

    return predicted_data

# Main Execution
def main():
    logging.info("Fetching data from PostgreSQL...")
    df = fetch_sensor_data()
    logging.info("Data fetched successfully.")

    logging.info("Starting anomaly detection...")
    detect_anomalies(df)

    logging.info("Preprocessing data for LSTM...")
    scaled_data, scaler = preprocess_data(df)

    logging.info("Creating sequences for training...")
    look_back = 30  # Use last 30 data points to predict the next
    X, y = create_sequences(scaled_data, look_back)

    # Check for NaN values before training
    if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
        logging.error("NaN values detected in training data. Exiting...")
        return

    # Step 5: Split the data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logging.info("Training LSTM model...")
    model, history = train_lstm_model(X_train, y_train)

    logging.info("Evaluating model performance...")
    predicted_data = evaluate_lstm_model(model, X_test, y_test, scaler)

    logging.info("Forecasting complete.")

if __name__ == "__main__":
    main()

