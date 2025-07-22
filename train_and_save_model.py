import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Check current directory
print("Current directory:", os.getcwd())

# Load the dataset
df = pd.read_csv("creditcard.csv")  # Make sure this file exists in the same folder

# Prepare data
X = df.drop("Class", axis=1)
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "random_forest_fraud_model.pkl")
print("✅ Model saved as 'random_forest_fraud_model.pkl'")
