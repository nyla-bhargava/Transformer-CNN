import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from model import Stage2Model
from dataset import OffTargetDataset
from utils import set_seed


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proxy_df = pd.read_csv(args.proxy_csv)

    train_df, val_df = train_test_split(
        proxy_df,
        test_size=0.15,
        stratify=proxy_df["label"],
        random_state=42
    )

    all_gRNAs = proxy_df.gRNA.unique()
    gRNA_to_idx = {g: i for i, g in enumerate(all_gRNAs)}

    sg_embeddings = torch.zeros(len(all_gRNAs), args.sg_dim)

    max_len = max(
        proxy_df.gRNA.str.len().max(),
        proxy_df.OT.str.len().max()
    )

    train_ds = OffTargetDataset(
        train_df, sg_embeddings, gRNA_to_idx, max_len
    )
    val_ds = OffTargetDataset(
        val_df, sg_embeddings, gRNA_to_idx, max_len
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = Stage2Model(args.sg_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            logits = model(
                batch["pair"],
                batch["mv"],
                batch["pam"],
                batch["sg_emb"]
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed")

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
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

        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_stage2.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sg_dim", type=int, default=768)
    args = parser.parse_args()

    main(args)
