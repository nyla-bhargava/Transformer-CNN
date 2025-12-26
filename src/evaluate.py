from utils import set_seed
set_seed(42)
import torch
from torch.utils.data import DataLoader
from dataset import OffTargetDataset
from model import CNNOnly, TransformerOnly, TransformerCNN
from sklearn.metrics import roc_auc_score
import pandas as pd

def evaluate_model(path, model_class, df_path):
    df = pd.read_csv(df_path)
    ds = OffTargetDataset(df, device)
    loader = DataLoader(ds, batch_size=32)

    model = model_class().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze().cpu().numpy()
            preds.extend(logits)
            labels.extend(y.cpu().numpy())

    return roc_auc_score(labels, preds)
