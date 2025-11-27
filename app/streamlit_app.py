import streamlit as st
from src.inference.predict import predict_single

st.title("Credit Card Fraud Detector")

time = st.number_input("Time")
amount = st.number_input("Amount")
features = {}

for i in range(1, 29):
    features[f"V{i}"] = st.number_input(f"V{i}")

if st.button("Predict"):
    features["Time"] = time
    features["Amount"] = amount
    prob = predict_single(features)
    st.success(f"Fraud probability: {prob:.4f}")
