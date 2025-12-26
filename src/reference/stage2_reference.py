import os, random, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

TRUEOT_PATH = "/content/TrueOT_1806uniqueTriplet_gRNA_OT_label.csv"
PROXY_PATH  = "/content/Proxy_TrainCV.csv"

USE_STAGE1 = False   # False for ablation
EPOCHS = 30

proxy_df  = pd.read_csv(PROXY_PATH)
trueot_df = pd.read_csv(TRUEOT_PATH)

train_df, val_df = train_test_split(
    proxy_df,
    test_size=0.15,
    stratify=proxy_df["label"],
    random_state=42
)

STAGE1_MODEL = "zhihan1996/DNA_bert_6"
tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
stage1_model = AutoModel.from_pretrained(STAGE1_MODEL).to(device)
stage1_model.eval()
for p in stage1_model.parameters():
    p.requires_grad = False

@torch.no_grad()
def compute_sg_embeddings(seqs, batch_size=64):
    embs = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = stage1_model(**toks)
        embs.append(out.last_hidden_state[:,0,:].cpu())
    return torch.cat(embs, dim=0)

all_gRNAs = pd.concat([train_df.gRNA, val_df.gRNA, trueot_df.gRNA]).unique()
gRNA_to_idx = {g:i for i,g in enumerate(all_gRNAs)}

if USE_STAGE1:
    sg_embeddings = compute_sg_embeddings(list(all_gRNAs))
    torch.save(sg_embeddings, "sg_embeddings.pt")
    sg_dim = sg_embeddings.shape[1]
else:
    sg_embeddings = torch.zeros(len(all_gRNAs), 768)
    sg_dim = 768

NUC2IDX = {'A':0,'C':1,'G':2,'T':3}

def one_hot(seq):
    arr = np.zeros((len(seq),4),dtype=np.float32)
    for i,b in enumerate(seq):
        if b in NUC2IDX:
            arr[i,NUC2IDX[b]] = 1
    return arr

def mismatch_vec(sg,ot):
    L=min(len(sg),len(ot))
    return np.array([0 if sg[i]==ot[i] else 1 for i in range(L)],dtype=np.float32)

def pam_distance_encoding(L, alpha=0.3):
    d = np.arange(L)[::-1]
    return np.exp(-alpha * d).astype(np.float32)

MAX_LEN = max(proxy_df.gRNA.str.len().max(), proxy_df.OT.str.len().max())

class OffTargetDataset(Dataset):
    def __init__(self, df):
        self.df=df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self,i):
        r=self.df.iloc[i]
        sg,ot=r.gRNA,r.OT

        sg_oh=one_hot(sg)
        ot_oh=one_hot(ot)
        mv=mismatch_vec(sg,ot)
        pam=pam_distance_encoding(len(mv))

        def pad(x):
            return np.pad(x,((0,MAX_LEN-len(x)),(0,0)))

        pair=np.concatenate([pad(sg_oh),pad(ot_oh)],axis=1)
        mv = np.pad(mv,(0,MAX_LEN-len(mv)))
        pam= np.pad(pam,(0,MAX_LEN-len(pam)))

        return {
            "pair": torch.tensor(pair),
            "mv": torch.tensor(mv),
            "pam": torch.tensor(pam),
            "sg_emb": sg_embeddings[gRNA_to_idx[sg]],
            "label": torch.tensor(float(r.label))
        }

class Stage2Model(nn.Module):
    def __init__(self, sg_dim):
        super().__init__()
        self.input_proj = nn.Linear(10,128)
        self.cnn = nn.Sequential(
            nn.Conv1d(128,128,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(128,128,3,padding=1),
            nn.ReLU()
        )
        enc = nn.TransformerEncoderLayer(128,4,batch_first=True)
        self.tr = nn.TransformerEncoder(enc,2)
        self.sg_proj = nn.Linear(sg_dim,128)
        self.dropout = nn.Dropout(0.3)
        self.cls = nn.Linear(256,1)

    def forward(self,pair,mv,pam,sg_emb):
        x=torch.cat([pair,mv.unsqueeze(-1),pam.unsqueeze(-1)],dim=-1)
        x=self.input_proj(x)
        x=self.cnn(x.transpose(1,2)).transpose(1,2)
        x=self.tr(x)
        pooled=x.mean(1)
        fused=torch.cat([pooled,self.sg_proj(sg_emb)],1)
        fused=self.dropout(fused)
        return self.cls(fused).squeeze(-1)

model = Stage2Model(sg_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)

#Training
def run_epoch(loader,train):
    model.train() if train else model.eval()
    ys,ps,ls=[],[],0
    for b in loader:
        for k in b: b[k]=b[k].to(device)
        if train: optimizer.zero_grad()
        logits=model(b["pair"],b["mv"],b["pam"],b["sg_emb"])
        loss=criterion(logits,b["label"])
        if train:
            loss.backward()
            optimizer.step()
        ls+=loss.item()*len(b["label"])
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(b["label"].cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    return ls/len(y), roc_auc_score(y,p), average_precision_score(y,p)

train_loader = DataLoader(OffTargetDataset(train_df),64,shuffle=True)
val_loader   = DataLoader(OffTargetDataset(val_df),128)
test_loader  = DataLoader(OffTargetDataset(trueot_df),128)

best_auc=0
for e in range(EPOCHS):
    _, ta, _ = run_epoch(train_loader,True)
    _, va, _ = run_epoch(val_loader,False)
    print(f"Epoch {e+1:02d} | Train AUC {ta:.4f} | Val AUC {va:.4f}")
    if va>best_auc:
        best_auc=va
        torch.save(model.state_dict(),"best_stage2.pt")

#Evaluation
model.load_state_dict(torch.load("best_stage2.pt"))
model.eval()

with torch.no_grad():
    _, test_auc, test_aupr = run_epoch(test_loader,False)

print("\n=== TRUEOT GENERALIZATION ===")
print(f"AUC  : {test_auc:.4f}")
print(f"AUPR : {test_aupr:.4f}")

@torch.no_grad()
def mc_dropout(loader,T=30):
    model.train()
    preds=[]
    for _ in range(T):
        batch=[]
        for b in loader:
            for k in b: b[k]=b[k].to(device)
            batch.append(torch.sigmoid(model(b["pair"],b["mv"],b["pam"],b["sg_emb"])).cpu().numpy())
        preds.append(np.concatenate(batch))
    model.eval()
    preds=np.stack(preds)
    return preds.mean(0), preds.std(0)

mean_pred, std_pred = mc_dropout(test_loader)

np.save("trueot_pred_mean.npy",mean_pred)
np.save("trueot_pred_uncertainty.npy",std_pred)

torch.save(torch.load("best_stage2.pt"),
           "stage2_with_stage1.pth" if USE_STAGE1 else "stage2_no_stage1.pth")
