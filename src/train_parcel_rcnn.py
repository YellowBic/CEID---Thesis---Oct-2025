# /content/drive/MyDrive/parcel_centric/src/train_parcel_rcnn.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np, pandas as pd

from dataset_parcel_rcnn import ParcelChipsRCNN

class CNNBackbone(nn.Module):
    """Small convnet that maps (B,C,H,W) -> (B, D) via GAP."""
    def __init__(self, in_ch=7, width=48, feat_dim=256):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(c, 2*c, 3, 1, 1, bias=False), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, 1, 1, bias=False), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(2*c, 4*c, 3, 1, 1, bias=False), nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(4*c, feat_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)     # (B, 4c)
        x = self.proj(x)               # (B, D)
        return x

class RCNNModel(nn.Module):
    """
    Expects x: (B, T, C, H, W)  where T=10, C=6 or 7
    """
    def __init__(self, in_ch=7, n_classes=20, feat_dim=256, gru_hidden=256, bidirectional=True, num_layers=1, dropout=0.1):
        super().__init__()
        self.backbone = CNNBackbone(in_ch=in_ch, width=48, feat_dim=feat_dim)
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout= (dropout if num_layers > 1 else 0.0)
        )
        out_dim = gru_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes)
        )

    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        feat = self.backbone(x)          # (B*T, D)
        feat = feat.view(B, T, -1)       # (B, T, D)
        seq, _ = self.gru(feat)          # (B, T, H*)
        last = seq[:, -1, :]             # last timestep
        logits = self.head(last)         # (B, n_classes)
        return logits

@torch.no_grad()
def evaluate(model, loader, device="cuda", n_classes=20):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    top3_hits, total = 0, 0
    for x, y in loader:
        # x: (T,C,H,W) -> (B,T,C,H,W)
        x = x.to(device)                         # (B,T,C,H,W)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        for yt, yp in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[yt, yp] += 1
        top3 = logits.topk(3, dim=1).indices
        top3_hits += (top3.eq(y[:, None])).any(1).sum().item()
        total += y.size(0)
    acc = cm.trace() / cm.sum().clip(min=1)
    # macro-F1
    f1 = []
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1.append(2 * prec * rec / (prec + rec + 1e-9))
    macro_f1 = float(np.mean(f1))
    top3_acc = top3_hits / max(1, total)
    return {"acc": float(acc), "macro_f1": macro_f1, "cm": cm, "top3_acc": top3_acc}

def fit_rcnn(index_csv_train,
             index_csv_val,
             in_ch=7,
             n_classes=20,
             bs=16,
             lr=3e-4,
             epochs=30,
             num_workers=2,
             class_weights=None,
             oversample=True,
             out_dir=None):
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ParcelChipsRCNN(index_csv_train, normalize=True, augment=True,  add_mask_channel=(in_ch==7))
    val_ds   = ParcelChipsRCNN(index_csv_val,   normalize=True, augment=False, add_mask_channel=(in_ch==7))

    # Oversampling based on class freq
    sampler = None
    if oversample:
        tr = pd.read_csv(index_csv_train)
        counts = tr["class_id"].value_counts().to_dict()
        w_class = {k: 1.0 / np.sqrt(v) for k, v in counts.items()}
        weights = tr["class_id"].map(w_class).astype("float32").values
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
                          num_workers=num_workers, pin_memory=(device=="cuda"))
    val_ld   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                          num_workers=num_workers, pin_memory=(device=="cuda"))

    model = RCNNModel(in_ch=in_ch, n_classes=n_classes).to(device)
    crit  = nn.CrossEntropyLoss(weight=class_weights)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    is_cuda = (device=="cuda")
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    # --- logging setup (όπως στο CNN) ---
    if out_dir is None:
        # infer ROOT από το path του index (…/parcel_centric/indices/parcel_train.csv -> …/parcel_centric)
        root = os.path.dirname(os.path.dirname(os.path.abspath(index_csv_train)))
        out_dir = os.path.join(root, "outputs", "train_rcnn")
    os.makedirs(out_dir, exist_ok=True)
    log_csv = os.path.join(out_dir, "log.csv")
    if not os.path.exists(log_csv):
        # πρώτη γραμμή ως "header row" (ίδιο μοτίβο με CNN)
        pd.DataFrame([["epoch","train_loss","val_acc","val_macro_f1","val_top3","lr"]]).to_csv(
            log_csv, index=False, header=False
        )

    # --- train loop ---
    best, best_state = -1.0, None
    for e in range(1, epochs+1):
        model.train(); run = 0.0
        for x, y in train_ld:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=is_cuda, dtype=amp_dtype):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()

        train_loss = run / max(1, len(train_ld))
        m = evaluate(model, val_ld, device=device, n_classes=n_classes)
        sched.step()

        lr_now = opt.param_groups[0]['lr']
        print(f"Epoch {e:03d} | loss {train_loss:.3f} | val acc {m['acc']:.3f} | "
              f"val macroF1 {m['macro_f1']:.3f} | top3 {m['top3_acc']:.3f}")

        # --- append log row ---
        with open(log_csv, "a") as f:
            f.write(f"{e},{train_loss:.6f},{m['acc']:.6f},{m['macro_f1']:.6f},{m['top3_acc']:.6f},{lr_now:.6e}\n")

        # κρατάμε το καλύτερο macro-F1
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print("RCNN log saved to:", log_csv)
    return model
