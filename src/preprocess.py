"""
Preprocess CRISPR Off-Target Datasets using DNABERT

This script converts raw CSV datasets into pooled DNABERT embeddings
used for training and evaluation.

Expected input schema:
- gRNA: nucleotide sequence (string)
- label: binary off-target label (0/1)
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILES = {
    "proxy_train": "Proxy_TrainCV.csv",
    "proxy_val": "Proxy_Validation.csv",
    "trueot": "TrueOT_1806uniqueTriplet_gRNA_OT_label.csv"
}

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load DNABERT
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(DEVICE)

model.eval()
for p in model.parameters():
    p.requires_grad = False

# -------------------------
# Encoding function
# -------------------------
def encode_sequences(seqs):
    embeddings = []

    for i in tqdm(range(0, len(seqs), BATCH_SIZE)):
        batch = seqs[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (outputs * mask).sum(1) / mask.sum(1)
        embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)

# -------------------------
# Main preprocessing loop
# -------------------------
for name, fname in FILES.items():
    path = os.path.join(RAW_DIR, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing raw dataset: {path}\n"
            f"Please place the file in {RAW_DIR} as described in README."
        )

    df = pd.read_csv(path)

    # Schema validation
    if "gRNA" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{fname} must contain columns: 'gRNA' and 'label'"
        )

    print(f"\nProcessing {fname}")
    print("Samples:", len(df))
    print("Positive ratio:", df["label"].mean())

    X = encode_sequences(df["gRNA"].tolist())
    y = df["label"].values.astype(np.int64)

    out_path = os.path.join(OUT_DIR, f"{name}_encoded.npz")
    np.savez_compressed(out_path, X=X, y=y)

    print(f"Saved: {out_path}")
    print("Embedding shape:", X.shape)

print("\nPreprocessing completed successfully.")
