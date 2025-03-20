import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

# Load dataset
df = pd.read_csv("cardekho_dataset.csv")

# Feature Engineering: Creating new features
df['car_age'] = 2025 - df['year']  # Convert 'year' to 'car_age'
df['km_driven_log'] = np.log1p(df['km_driven'])  # Log transformation to reduce skew

# Drop unnecessary columns
df.drop(columns=['year', 'name'], inplace=True)  # 'year' replaced by 'car_age', 'name' is non-numeric

# Feature Selection: Select top 10 important features
X = df.drop(columns=['selling_price'])
y = df['selling_price']

selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Keep only selected features
df = df[selected_features]
df['selling_price'] = y  # Add target variable back

# Save processed data
df.to_csv("processed_cardekho_dataset.csv", index=False)

print("Feature Selection Completed. Processed dataset saved!")
