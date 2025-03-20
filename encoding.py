import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load dataset
file_path = "cardekho_dataset.csv"  # Adjust the path as needed
df = pd.read_csv(file_path)

# Initialize OneHotEncoder with the correct argument
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Fixed argument

# Fit and transform categorical features
encoded_features = encoder.fit_transform(df[['fuel_type', 'transmission_type']])

# Convert to DataFrame and add back to df
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
df = df.join(encoded_df)

# Drop original categorical columns
df.drop(columns=['fuel_type', 'transmission_type'], inplace=True)
# Frequency Encoding for brand & model
df['brand_encoded'] = df['brand'].map(df['brand'].value_counts())
df['model_encoded'] = df['model'].map(df['model'].value_counts())

# Drop original brand and model columns
df.drop(columns=['brand', 'model'], inplace=True)


# Save the processed dataset (optional)
df.to_csv("processed_cardekho_dataset.csv", index=False)

# Display the first few rows
print(df.head())

# Save the cleaned dataset
df.to_csv("processed_cardekho_dataset.csv", index=False)

print("Processed dataset saved as 'processed_cardekho_dataset.csv'")

