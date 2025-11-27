import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            probs = torch.sigmoid(model(Xb)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(yb.numpy())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob
