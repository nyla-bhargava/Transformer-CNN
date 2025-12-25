"""
Reproduce Results for Transformer–CNN CRISPR Off-Target Prediction

This script reproduces the main and ablation results reported in the paper.
It evaluates pretrained models on the TrueOT benchmark.
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

MODELS = {
    "CNN-only (DNABERT)": {
        "class": CNNOnly,
        "ckpt": "../checkpoints/cnn_only.pt"
    },
    "Transformer-only (DNABERT)": {
        "class": TransformerOnly,
        "ckpt": "../checkpoints/transformer_only.pt"
    },
    "Transformer–CNN (ours)": {
        "class": TransformerCNN,
        "ckpt": "../checkpoints/transformer_cnn.pt"
    }
}

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze()
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return roc_auc_score(labels, preds)

def main():
    df = pd.read_csv(TRUEOT_CSV)
    dataset = OffTargetDataset(df, DEVICE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    results = []

    for name, cfg in MODELS.items():
        model = cfg["class"]().to(DEVICE)
        model.load_state_dict(torch.load(cfg["ckpt]()
