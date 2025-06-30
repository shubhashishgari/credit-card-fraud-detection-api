from dotenv import load_dotenv
import os

load_dotenv()

import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# FastAPI instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is live."}

# Load environment configs
DATA_PATH = os.getenv("DATA_PATH", "archive/creditcard.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

@app.post("/predict")
def predict_fraud(data: Dict[str, float]):
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        raise HTTPException(status_code=500, detail="Required model or scaler not found.")

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)

        df = pd.DataFrame([data])[features]
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 6),
            "result": "Fraud" if prediction == 1 else "Not Fraud"
        }

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing or unexpected feature: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
