import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv('data.csv')
# Data Preprocessing
data = data.dropna()  # Drop rows with missing values

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
features = data.drop('MSRP', axis=1)
target = data['MSRP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model, scaler, and feature names
with open('car_price_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save feature names
with open('features.pkl', 'wb') as features_file:
    pickle.dump(list(features.columns), features_file)
