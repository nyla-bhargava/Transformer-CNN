import argparse
import os
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from model import build_model
from dataset import load_trueot
from utils import set_seed

def evaluate(model, dataloader, device):
    y_true, y_score = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = model(x).squeeze().cpu()
            y_score.extend(preds.tolist())
            y_true.extend(y.tolist())
    return y_true, y_score

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(os.path.join(cfg["output"]["checkpoint_dir"], "best.pt"))
    model.load_state_dict(ckpt)

    dataloader = load_trueot(cfg["data"]["test_csv"])
    y_true, y_score = evaluate(model, dataloader, device)

    roc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr = auc(recall, precision)

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Save table
    df = pd.DataFrame([{
        "ROC_AUC": roc,
        "PR_AUC": pr
    }])
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

    # ROC plot
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.savefig(os.path.join(results_dir, "pr_curve.png"))

    print(f"ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
