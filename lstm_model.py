import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Check if GPU is available
print("🔍 TensorFlow Device Check:", tf.config.list_physical_devices('GPU'))

# Load dataset
df = pd.read_csv("data/cleaned_ILINet.csv")  # Ensure correct file path
print("✅ Dataset Loaded Successfully!")

# Print column names to confirm structure
print("🔍 Columns in dataset:", df.columns)

# Convert YEAR column to datetime (Ensure correct column name)
date_column = 'YEAR' if 'YEAR' in df.columns else 'year'
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

# Feature selection (Ensure correct column name for ILI)
ili_column = '% WEIGHTED ILI' if '% WEIGHTED ILI' in df.columns else df.columns[3]  # Adjust index if needed
data = df[ili_column].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create sequences
SEQ_LENGTH = 10  # Adjust as needed
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split into training and testing sets (80-20 split)
SPLIT_INDEX = int(len(X) * 0.8)
X_train, X_test = X[:SPLIT_INDEX], X[SPLIT_INDEX:]
y_train, y_test = y[:SPLIT_INDEX], y[SPLIT_INDEX:]

print(f"✅ Data Prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display Model Summary
print("📌 Model Summary:")
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
print("🚀 Training Model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Save Model
model.save("lstm_model.h5")
print("✅ Model Training Complete! Model Saved as 'lstm_model.h5'.")

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LSTM Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
