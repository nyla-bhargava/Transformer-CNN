import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
dnabert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")

def encode_with_dnabert(seqs, device):
    all_embeddings = []
    for seq in seqs:
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = dnabert(**inputs).last_hidden_state
            attn_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (out * attn_mask).sum(1) / attn_mask.sum(1)
        all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings)

class OffTargetDataset(Dataset):
    def __init__(self, df, device):
        self.device = device
        self.seqs = df["gRNA"].tolist()
        self.labels = df["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = encode_with_dnabert([self.seqs[idx]], self.device)[0]
        return emb, torch.tensor(self.labels[idx], dtype=torch.float32)
