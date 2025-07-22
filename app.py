import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

# Load dataset (skip broken rows)
try:
    df = pd.read_csv("creditcard.csv", on_bad_lines='skip')
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Load the trained fraud detection model
try:
    model = joblib.load("random_forest_fraud_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'fraud_model.pkl' exists in the 'model/' folder.")
    st.stop()

# Streamlit App UI
st.title("💳 Credit Card Fraud Detection")
st.markdown("Upload or input credit card transaction features to check if it's fraudulent.")

# Input features (adjust as per your model)
# Example features (V1 to V28, Amount)
# We'll just take a few for example, update based on your model input

input_data = {}
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']

for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=0.0)

if st.button("🔍 Detect Fraud"):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Show sample data
with st.expander("🔎 View Sample Data"):
    st.dataframe(df.head(10))
