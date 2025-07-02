import streamlit as st
import requests

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether it's **fraudulent** or **not**.")

# Define input features (must match model's expected inputs exactly)
features_list = [
    "Time", 
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28", "Amount"
]

# Collect inputs
input_data = {}
with st.form(key="fraud_form"):
    for feature in features_list:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.5f")
    
    submit = st.form_submit_button("🚀 Predict")

# Send request to API
if submit:
    api_url = "https://credit-card-fraud-detection-api.onrender.com/predict"
    
    with st.spinner("Sending data to the model and waiting for prediction..."):
        try:
            response = requests.post(api_url, json=input_data)
            result = response.json()

            if response.status_code == 200:
                st.success(f"🧾 Prediction: {result['result']}")
                st.write(f"🔍 Probability of fraud: `{result['probability']}`")
            else:
                st.error(f"❌ Error: {result.get('detail', 'Something went wrong')}")
        except Exception as e:
            st.error(f"⚠️ Failed to connect to API: {e}")
