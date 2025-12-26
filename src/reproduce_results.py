"""
Reproduce Ablation Results (Table 2)

This script reproduces the ablation study results reported in the paper:
CNN-only, Transformer-only, and Transformer–CNN models evaluated on TrueOT.
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import OffTargetDataset
from model import CNNOnly, TransformerOnly, TransformerCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRUEOT_CSV = "../data/processed/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv"
BATCH_SIZE = 32

MODELS = [
    ("CNN-only (DNABERT)", CNNOnly, "../checkpoints/cnn_only.pt"),
    ("Transformer-only (DNABERT)", TransformerOnly, "../checkpoints/transformer_only.pt"),
    ("Transformer–CNN (ours)", TransformerCNN, "../checkpoints/transformer_cnn.pt"),
]

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            y_pred.extend(probs.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    return roc_auc_score(y_true, y_pred)

def main():
    df = pd.read_csv(TRUEOT_CSV)
    dataset = OffTargetDataset(df, DEVICE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    rows = []

    for name, model_cls, ckpt in MODELS:
        model = model_cls().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        auc = evaluate(model, loader)

        rows.append({
            "Model": name,
            "Architecture": name.split("(")[0].strip(),
            "Dataset": "TrueOT",
            "ROC-AUC": round(auc, 4)
        })

        print(f"{name}: ROC-AUC = {auc:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv("../results/ablation_trueot.csv", index=False)

    print("\nAblation Study (Table 2):")
    print(results)

if __name__ == "__main__":
    main()
