# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge  # Using Ridge Regression (Better than plain Linear Regression)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("scaled_dataset.csv")

# Define features and target
X = df.drop(columns=['selling_price'])  # Independent variables
y = df['selling_price']  # Target variable

# Log transformation (if selling_price is skewed)
y = np.log1p(y)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge Regression Model (with regularization)
model = Ridge(alpha=1.0)  # Alpha helps prevent overfitting
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Reverse log transformation (if applied)
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Performance Metrics
print(f"📊 Model Performance (With Ridge Regression):")
print(f"MAE (Mean Absolute Error): {mae}")
print(f"MSE (Mean Squared Error): {mse}")
print(f"R² Score: {r2}")
