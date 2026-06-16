import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = 'data/fertilizer_data.csv'

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Please check the path.")
    exit()

df = pd.read_csv(file_path, encoding='utf-8')

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Print available columns
print("Available columns:", df.columns.tolist())

# Rename 'Soil_color' to 'Soil_Type' (if applicable)
if 'Soil_color' in df.columns:
    df.rename(columns={'Soil_color': 'Soil_Type'}, inplace=True)

# Define categorical columns
categorical_columns = ['District_Name', 'Soil_Type']

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.capitalize()  # Ensure consistent capitalization
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

        # Add 'unknown' category to handle unseen labels
        le.classes_ = np.append(le.classes_, "unknown")

        label_encoders[col] = le

# Save label encoders
encoder_file = 'models/label_encoders.pkl'
with open(encoder_file, 'wb') as f:
    pickle.dump(label_encoders, f)

print(f"Encoders saved successfully in '{encoder_file}'")

# Define features and target variable
X = df.drop(columns=['Fertilizer'])
y = df['Fertilizer']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_file = 'models/standscaler.pkl'
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save trained model
model_file = 'models/fertilizer_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved successfully in '{model_file}'")
