import torch
import pandas as pd
from torch.utils.data import DataLoader

from model import Stage2Model
from dataset import OffTargetDataset
from utils import compute_metrics


def evaluate(trueot_csv, model_ckpt, sg_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(trueot_csv)
    all_gRNAs = df.gRNA.unique()
    gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}
    sg_embeddings = torch.zeros(len(all_gRNAs), sg_dim)

    max_len = max(df.gRNA.str.len().max(), df.OT.str.len().max())

    ds = OffTargetDataset(df, sg_embeddings, gRNA_to_idx, max_len)
    loader = DataLoader(ds, batch_size=128)

    model = Stage2Model(sg_dim).to(device)
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            out = torch.sigmoid(
                model(
                    batch["pair"],
                    batch["mv"],
                    batch["pam"],
                    batch["sg_emb"]
                )
            )
            preds.append(out.cpu())
            labels.append(batch["label"].cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    return compute_metrics(labels, preds)
