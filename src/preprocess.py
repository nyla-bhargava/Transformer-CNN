import numpy as np
from dataset import encode_with_dnabert

def save_embeddings(df, out_path):
    seqs = df["gRNA"].tolist()
    emb = encode_with_dnabert(seqs, device)
    np.savez_compressed(out_path, emb=emb, labels=df["label"].values)
