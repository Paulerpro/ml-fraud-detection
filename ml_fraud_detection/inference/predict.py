import torch
import numpy as np
import pickle
import pandas as pd

from ml_fraud_detection.training.model import MLP

device = torch.device("cpu")

model = MLP(input_dim=30)

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pt")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

scaler = pickle.load(open(SCALER_PATH, "rb"))

def predict_fraud(feature_dict):
    df = pd.DataFrame([feature_dict])
    df_scaled = scaler.transform(df)
    X = torch.tensor(df_scaled.values, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()
    
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(prob >= 0.5)
    }

