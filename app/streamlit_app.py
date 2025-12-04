import streamlit as st
import requests

API_URL = "https://ml-fraud-detection-backend.onrender.com/predict"


st.title("Fraud Detection System")
st.write("Enter the 30 features used for prediction")

time = st.number_input("Time")
amount = st.number_input("Amount")
features = {}

for i in range(1, 29):
    features[f"V{i}"] = st.number_input(f"V{i}")

if st.button("Predict Fraud"):
    features["Time"] = time
    features["Amount"] = amount
    payload = {"features": features}

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.write("### Fraud Probability:", result["fraud_probability"])
        st.write("### Fraud:", result["is_fraud"])
    else:
        st.error("API Error: Could not get prediction")
