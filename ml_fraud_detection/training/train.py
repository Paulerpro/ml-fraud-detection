import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import mlflow
import mlflow.pytorch
from ml_fraud_detection.training.model import MLP
from ml_fraud_detection.training.dataset import CreditCardDataset
from ml_fraud_detection.training.utils import train_epoch, evaluate


def main():

    mlflow.set_experiment("fraud_detection_experiment")
    with mlflow.start_run():
    
        SEED = 42
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mlflow.log_param("device", str(device))

        df = pd.read_csv("creditcard.csv")
        features = [c for c in df.columns if c != "Class"]
        X = df[features].copy()
        y = df["Class"].values

        scaler = StandardScaler()
        X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])

        mlflow.log_param("features_dim", X.shape[1])

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
        )

        X_train, X_val, X_test = X_train.values.astype(np.float32), 
        X_val.values.astype(np.float32), 
        X_test.values.astype(np.float32)

        train_loader = DataLoader(CreditCardDataset(X_train, y_train), batch_size=1024, shuffle=True)
        val_loader   = DataLoader(CreditCardDataset(X_val, y_val), batch_size=1024)
        test_loader  = DataLoader(CreditCardDataset(X_test, y_test), batch_size=1024)

        model = MLP(input_dim=X_train.shape[1]).to(device)
        mlflow.log_param("model_type", "MLP")

        counts = np.bincount(y_train)
        pos_weight = torch.tensor(counts[0] / counts[1]).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("weight_decay", 1e-5)
        mlflow.log_param("batch_size", 1024)
        mlflow.log_param("epochs", 20)
        mlflow.log_param("pos_weight", float(counts[0] / counts[1]))

        best_auc = 0
        patience, wait = 5, 0

        for epoch in range(20):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            yv, pv = evaluate(model, val_loader, device)

            auc = roc_auc_score(yv, pv)
            ap  = average_precision_score(yv, pv)

            print(f"Epoch {epoch+1} | Loss={loss:.4f} AUC={auc:.4f} AP={ap:.4f}")

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_auc", auc, step=epoch)
            mlflow.log_metric("val_ap", ap, step=epoch)

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "models/best_model.pt")
                pickle.dump(scaler, open("models/scaler.pkl", "wb"))
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping...")
                    break

        yt, pt = evaluate(model, test_loader, device)
        test_auc = roc_auc_score(yt, pt)
        test_ap  = average_precision_score(yt, pt)
    
        print("Test AUC:", test_auc)
        print("Test AP :", test_ap)

        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_ap", test_ap)

        mlflow.pytorch.log_model(
            model, artifact_path="model"
        )

        mlflow.log_artifact("models/scaler.pkl", artifact_path="preprocessing")

        print("MLflow run completed!")

if __name__ == "__main__":
    main()
