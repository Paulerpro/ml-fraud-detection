import torch
import numpy as np
import pickle
import pandas as pd

from ml_fraud_detection.training.model import MLP

device = torch.device("cpu")

# Load model + scaler
model = MLP(input_dim=30)

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ml-fraud-detection/

# Step 2: Path to the model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pt")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# model.load_state_dict(torch.load("models/model.pt", map_location=device))
model.eval()

# SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
scaler = pickle.load(open(SCALER_PATH, "rb"))

def predict_single(feature_dict):
    df = pd.DataFrame([feature_dict])
    df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
    X = torch.tensor(df.values, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()

    return float(prob)
