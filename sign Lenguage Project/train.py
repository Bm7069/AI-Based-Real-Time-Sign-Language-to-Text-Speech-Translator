# train.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KeypointSequenceDataset
from model import TemporalTransformerClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from torchmetrics.classification import MulticlassAccuracy

ROOT = "dataset"
SEQ_LEN=30
BATCH=16
EPOCHS=30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ds = KeypointSequenceDataset(ROOT, seq_len=SEQ_LEN)
# split
indices = list(range(len(ds)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
from torch.utils.data import Subset
train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH, shuffle=False)

num_classes = len(ds.classes())
model = TemporalTransformerClassifier(input_dim=21*3, num_classes=num_classes).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
acc = MulticlassAccuracy(num_classes=num_classes).to(DEVICE)

for epoch in range(EPOCHS):
    model.train()
    tot_loss=0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()*xb.size(0)
    avg_loss = tot_loss/len(train_loader.dataset)
    # val
    model.eval(); val_acc=0; val_cnt=0
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1)
            val_acc += (preds==yb).sum().item()
            val_cnt += xb.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_acc={val_acc/val_cnt:.4f}")
# save
torch.save({'model':model.state_dict(),'classes':ds.classes()}, "model.pth")
