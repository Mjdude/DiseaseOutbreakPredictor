import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load cleaned dataset
df = pd.read_csv("data/cleaned_ILINet.csv")

# Ensure date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Use the 'date' column as the X-axis (convert to numerical values)
df['timestamp'] = df['date'].map(pd.Timestamp.toordinal)

# Select features and target variable (adjust column name as needed)
X = df[['timestamp']]  # Time-based feature
y = df['% WEIGHTED ILI']  # Target: Influenza-like illness percentage

# Split data into training & testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(df['timestamp'], y, label="Actual Data", color="blue", alpha=0.5)
plt.plot(X_test, y_pred, label="Predicted Trend", color="red")
plt.xlabel("Date")
plt.ylabel("% Weighted ILI")
plt.title("Influenza Trend Prediction")
plt.legend()
plt.show()
