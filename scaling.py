import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load processed dataset
file_path = "processed_cardekho_dataset.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# Define the features that need scaling (Ensure these columns exist in dataset)
features_to_scale = ['km_driven', 'mileage', 'engine', 'vehicle_age']

# Check if features exist in the dataset
missing_features = [feature for feature in features_to_scale if feature not in df.columns]
if missing_features:
    raise ValueError(f"Columns missing from dataset: {missing_features}")

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the features
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Save the processed dataset as 'scaled_dataset.csv'
df.to_csv("scaled_dataset.csv", index=False)

print("✅ Processed dataset saved as 'scaled_dataset.csv'.")
