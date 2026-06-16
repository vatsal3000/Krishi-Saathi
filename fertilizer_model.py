import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset (Ensure the correct path)
df = pd.read_csv("data/fertilizer_data.csv")


# Encoding categorical labels if needed
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoder for later use

# Splitting features and target variable
X = df.drop(columns=['Fertilizer'])  # Ensure correct column name
y = df['Fertilizer']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
with open("models/fertilizer_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model training complete. Files saved successfully.")
