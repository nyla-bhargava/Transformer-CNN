import torch
from utils import set_seed

set_seed(42)
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OffTargetDataset
from model import CNNOnly, TransformerOnly, TransformerCNN
from utils import compute_metrics
import pandas as pd

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    train_df = pd.read_csv(config["train_csv"])
    val_df = pd.read_csv(config["val_csv"])

    train_ds = OffTargetDataset(train_df, device)
    val_ds = OffTargetDataset(val_df, device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    if config["model"] == "cnn":
        model = CNNOnly().to(device)
    elif config["model"] == "transformer":
        model = TransformerOnly().to(device)
    else:
        model = TransformerCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    best_auc = 0
    for epoch in range(config["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        # validation AUC code here
        # save best model if val improves
