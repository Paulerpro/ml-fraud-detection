import torch
import numpy as np
import pickle
import pandas as pd

from src.training.model import MLP

device = torch.device("cpu")

# Load model + scaler
model = MLP(input_dim=30)
model.load_state_dict(torch.load("models/model.pt", map_location=device))
model.eval()

scaler = pickle.load(open("models/scaler.pkl", "rb"))

def predict_single(feature_dict):
    df = pd.DataFrame([feature_dict])
    df[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
    X = torch.tensor(df.values, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()

    return float(prob)
