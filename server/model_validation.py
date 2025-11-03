import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model_pytorch(model, test_X_path, test_y_path):
    model.eval()
    X = np.load(test_X_path)
    y = np.load(test_y_path)
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = model(X_t)
        preds = preds.argmax(1).numpy()
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    return {"accuracy": acc, "f1": f1}
