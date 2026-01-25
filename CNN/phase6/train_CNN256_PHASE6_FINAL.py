#!/usr/bin/env python
import os, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, log_loss, confusion_matrix
)

# =========================
# LIVE LOGGING
# =========================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =========================
# CONFIG (UNCHANGED)
# =========================
SEG_DIR = "/vol/projects/znoor/cnn_training/3class_exp/segments_256"
OUT_DIR = "/vol/projects/znoor/cnn_training/3class_exp_CNNBESTMODEL_Phase6"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 150
BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# =========================
# DATASET (UNCHANGED)
# =========================
class SegmentDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        x = d["windows"].astype(np.float32).reshape(256, -1)
        y = int(d["label"].item())
        pid = os.path.basename(self.files[idx]).split("_seg")[0]
        return torch.tensor(x), torch.tensor(y), pid

# =========================
# MODEL (UNCHANGED)
# =========================
class CNN256(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2375, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.flatten(1))

# =========================
# LOAD FILES (UNCHANGED)
# =========================
all_files = sorted(glob.glob(f"{SEG_DIR}/*.npz"))
pair_map = defaultdict(list)
for f in all_files:
    pid = os.path.basename(f).split("_seg")[0]
    pair_map[pid].append(f)

pairs = sorted(pair_map.keys())
np.random.seed(42)
np.random.shuffle(pairs)

n_train = int(0.6 * len(pairs))
train_pairs = set(pairs[:n_train])
val_pairs   = set(pairs[n_train:])

train_files = [f for p in train_pairs for f in pair_map[p]]
val_files   = [f for p in val_pairs   for f in pair_map[p]]

log("CNN256 Phase-6 FINAL")
log(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

# =========================
# CLASS WEIGHTS (UNCHANGED)
# =========================
labels_tmp = [int(np.load(f)["label"].item()) for f in train_files]
cnt = Counter(labels_tmp)
weights = np.array(
    [len(labels_tmp)/(cnt.get(i,1)*NUM_CLASSES) for i in range(NUM_CLASSES)],
    dtype=np.float32
)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(DEVICE))

# =========================
# LOADERS (UNCHANGED)
# =========================
train_loader = DataLoader(SegmentDataset(train_files), BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(SegmentDataset(val_files),   BATCH_SIZE, shuffle=False, num_workers=4)

# =========================
# METRIC HELPERS
# =========================
def pair_stats(votes, probs, true):
    c = Counter(votes)
    total = len(votes)
    pred = c.most_common(1)[0][0]
    agreement = c[pred]/total
    p = np.array([c.get(i,0)/total for i in range(NUM_CLASSES)])
    entropy = -np.sum(p*np.log(p+1e-9))
    margin = c.most_common(1)[0][1] - (c.most_common(2)[1][1] if len(c)>1 else 0)
    seg_acc = c.get(true,0)/total
    mean_conf = np.mean([pr[pred] for pr in probs])
    return pred, agreement, entropy, margin, seg_acc, mean_conf

def ece_score(probs, labels, bins=10):
    conf = np.max(probs,1)
    pred = np.argmax(probs,1)
    acc = pred==labels
    edges = np.linspace(0,1,bins+1)
    ece = 0.0
    for i in range(bins):
        m = (conf>edges[i])&(conf<=edges[i+1])
        if np.any(m):
            ece += abs(conf[m].mean()-acc[m].mean())*m.mean()
    return ece

# =========================
# TRAINING
# =========================
model = CNN256().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    log("="*70)
    log(f"EPOCH {epoch}/{EPOCHS}")
    log("="*70)

    for split, loader in [("train",train_loader),("val",val_loader)]:
        model.train() if split=="train" else model.eval()

        pair_votes = defaultdict(list)
        pair_probs = defaultdict(list)
        pair_truth = {}

        with torch.set_grad_enabled(split=="train"):
            for b,(x,y,pid) in enumerate(loader):
                x,y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                probs = torch.softmax(out,1)
                preds = probs.argmax(1)

                if split=="train":
                    loss = criterion(out,y)
                    opt.zero_grad(); loss.backward(); opt.step()

                for i,p in enumerate(pid):
                    pair_votes[p].append(preds[i].item())
                    pair_probs[p].append(probs[i].detach().cpu().numpy())

                    pair_truth[p]=y[i].item()

                if b%50==0:
                    log(f"[{split.upper()}] batch {b}/{len(loader)}")

        rows=[]
        pair_pred=[]; pair_true=[]; pair_prob_mean=[]

        log(f"[{split.upper()}][PAIR] summaries")
        for i,p in enumerate(sorted(pair_votes)):
            pred,agree,H,margin,seg_acc,conf = pair_stats(
                pair_votes[p], pair_probs[p], pair_truth[p]
            )
            pair_pred.append(pred)
            pair_true.append(pair_truth[p])
            pair_prob_mean.append(np.mean(pair_probs[p],0))

            rows.append({
                "pair_id":p,"true":pair_truth[p],"pred":pred,
                "agreement":agree,"entropy":H,
                "margin":margin,"seg_acc":seg_acc,"mean_conf":conf
            })

            log(f"[{split.upper()}][PAIR {i+1:03d}] agree={agree:.2f} H={H:.2f} conf={conf:.2f}")

        pd.DataFrame(rows).to_csv(
            f"{OUT_DIR}/pair_summary_{split}_epoch_{epoch:03d}.csv",
            index=False
        )

        probs_np = np.array(pair_prob_mean)
        labels_np = np.array(pair_true)

        acc = accuracy_score(labels_np,pair_pred)
        bal = balanced_accuracy_score(labels_np,pair_pred)
        pM  = precision_score(labels_np,pair_pred,average="macro",zero_division=0)
        rM  = recall_score(labels_np,pair_pred,average="macro",zero_division=0)
        fM  = f1_score(labels_np,pair_pred,average="macro")
        fMi = f1_score(labels_np,pair_pred,average="micro")
        fW  = f1_score(labels_np,pair_pred,average="weighted")
        mcc = matthews_corrcoef(labels_np,pair_pred)
        kap = cohen_kappa_score(labels_np,pair_pred)
        roc = roc_auc_score(labels_np,probs_np,multi_class="ovr",average="macro")
        nll = log_loss(labels_np,probs_np)
        brier = np.mean(np.sum((probs_np-np.eye(NUM_CLASSES)[labels_np])**2,1))
        ece = ece_score(probs_np,labels_np)
        cm = confusion_matrix(labels_np,pair_pred)

        log(f"[{split.upper()}] Accuracy          : {acc:.4f}")
        log(f"[{split.upper()}] Balanced Accuracy : {bal:.4f}")
        log(f"[{split.upper()}] Macro Precision   : {pM:.4f}")
        log(f"[{split.upper()}] Macro Recall      : {rM:.4f}")
        log(f"[{split.upper()}] Macro F1          : {fM:.4f}")
        log(f"[{split.upper()}] Micro F1          : {fMi:.4f}")
        log(f"[{split.upper()}] Weighted F1       : {fW:.4f}")
        log(f"[{split.upper()}] MCC               : {mcc:.4f}")
        log(f"[{split.upper()}] Cohen Kappa       : {kap:.4f}")
        log(f"[{split.upper()}] ROC-AUC (macro)   : {roc:.6f}")
        log(f"[{split.upper()}] NLL (log-loss)    : {nll:.6f}")
        log(f"[{split.upper()}] Brier score       : {brier:.4f}")
        log(f"[{split.upper()}] ECE (10 bins)     : {ece:.4f}")
        log(f"[{split.upper()}] Confusion Matrix:")
        for r in cm:
            log(f"[{split.upper()}] {r.tolist()}")

        np.savetxt(
            f"{OUT_DIR}/confusion_matrix_{split}_epoch_{epoch:03d}.txt",
            cm, fmt="%d"
        )

    torch.save({"epoch":epoch,"model":model.state_dict()},
               f"{OUT_DIR}/epoch_{epoch:03d}.pt")

