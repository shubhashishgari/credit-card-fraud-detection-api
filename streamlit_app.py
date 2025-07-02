import streamlit as st
import requests
import json

# Title
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether it's fraud or not.")

# Define all required fields — same as your model expects
input_data = {}

# You can adjust fields based on features.pkl contents
features_list = [
    "Time", 
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28", "Amount"
]


# Input form
with st.form(key='fraud_form'):
    for feature in features_list:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    submit = st.form_submit_button("Predict")

# Predict
if submit:
    api_url = "https://<your-render-url>.onrender.com/predict"  # replace with your actual URL
    try:
        response = requests.post(api_url, json=input_data)
        result = response.json()

        if response.status_code == 200:
            st.success(f"Prediction: {result['result']}")
            st.write(f"Probability of fraud: {result['probability']}")
        else:
            st.error(f"Error: {result.get('detail', 'Something went wrong')}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
