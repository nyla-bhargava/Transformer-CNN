import numpy as np
import torch
from torch.utils.data import Dataset


NUC2IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in NUC2IDX:
            arr[i, NUC2IDX[b]] = 1
    return arr


def mismatch_vec(sg, ot):
    L = min(len(sg), len(ot))
    return np.array(
        [0 if sg[i] == ot[i] else 1 for i in range(L)],
        dtype=np.float32
    )


def pam_distance_encoding(L, alpha=0.3):
    d = np.arange(L)[::-1]
    return np.exp(-alpha * d).astype(np.float32)


class OffTargetDataset(Dataset):
    def __init__(self, df, sg_embeddings, gRNA_to_idx, max_len):
        self.df = df.reset_index(drop=True)
        self.sg_embeddings = sg_embeddings
        self.gRNA_to_idx = gRNA_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _pad(self, x, width):
        return np.pad(x, width, mode="constant")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sg, ot = row.gRNA, row.OT

        sg_oh = one_hot(sg)
        ot_oh = one_hot(ot)

        mv = mismatch_vec(sg, ot)
        pam = pam_distance_encoding(len(mv))

        sg_oh = self._pad(sg_oh, ((0, self.max_len - len(sg_oh)), (0, 0)))
        ot_oh = self._pad(ot_oh, ((0, self.max_len - len(ot_oh)), (0, 0)))

        mv = self._pad(mv, (0, self.max_len - len(mv)))
        pam = self._pad(pam, (0, self.max_len - len(pam)))

        pair = np.concatenate([sg_oh, ot_oh], axis=1)

        return {
            "pair": torch.tensor(pair, dtype=torch.float32),
            "mv": torch.tensor(mv, dtype=torch.float32),
            "pam": torch.tensor(pam, dtype=torch.float32),
            "sg_emb": self.sg_embeddings[self.gRNA_to_idx[sg]],
            "label": torch.tensor(float(row.label))
        }
