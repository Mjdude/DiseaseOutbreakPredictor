import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load cleaned dataset
df = pd.read_csv("data/cleaned_ILINet.csv")

# Ensure date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Select feature for prediction (e.g., "% WEIGHTED ILI")
data = df["% WEIGHTED ILI"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Set sequence length
SEQ_LENGTH = 10  # Adjust as needed
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split into training (80%) and testing (20%)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("✅ Data Prepared for Training!")

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model with Early Stopping
print("🚀 Training Model...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

# Save Model
model.save("models/lstm_model.h5")
print("✅ Model Training Complete! Model Saved.")

# Plot Training Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
plt.legend()
plt.title("📉 LSTM Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
