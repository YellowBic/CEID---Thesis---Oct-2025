# /content/drive/MyDrive/parcel_centric/src/train_parcel_cls.py
import os, math, json, random, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from typing import Optional, Dict, Any

from dataset_parcel_cls import ParcelChipsCls

# ---------------------------
# Small helpers
# ---------------------------

def _seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _save_rng_state(path: str):
    state = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_random": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
        # convert tensors to lists for JSON
        state["torch_cuda_random"] = [t.cpu().tolist() for t in state["torch_cuda_random"]]
    with open(path, "w") as f:
        json.dump(state, f)

def _load_rng_state(path: str):
    with open(path, "r") as f:
        state = json.load(f)
    random.setstate(tuple(state["py_random"]))
    np.random.set_state(tuple(state["np_random"]))
    torch.set_rng_state(torch.tensor(state["torch_random"], dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda_random" in state:
        seq = [torch.tensor(s, dtype=torch.uint8, device="cuda") for s in state["torch_cuda_random"]]
        torch.cuda.set_rng_state_all(seq)

def _get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    # fallback: max epoch file
    cands = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pt")]
    if not cands:
        return None
    cands.sort()
    return os.path.join(ckpt_dir, cands[-1])

# ---------------------------
# Model
# ---------------------------
class ParcelCNN(nn.Module):
    def __init__(self, in_ch=61, n_classes=20, width=64):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),     nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64x64

            nn.Conv2d(c, 2*c, 3, 1, 1, bias=False),   nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.Conv2d(2*c,2*c, 3, 1, 1, bias=False),  nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32x32

            nn.Conv2d(2*c,4*c, 3, 1, 1, bias=False),  nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4*c, n_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)

# ---------------------------
# EMA (Exponential Moving Average of weights)
# ---------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = {k: v.to(next(iter(self.shadow.values())).device if self.shadow else "cpu")
                       for k, v in sd["shadow"].items()}

# ---------------------------
# Mixup / Cutmix
# ---------------------------
def _sample_lam(alpha: float) -> float:
    if alpha <= 0.0:
        return 1.0
    return np.random.beta(alpha, alpha)

def _apply_mixup_cutmix(x, y, mixup_alpha: float, cutmix_alpha: float):
    """Returns (x_aug, y, y_shuffled, lam, mode) where mode in {'mixup','cutmix',None}"""
    B, C, H, W = x.shape
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0

    if not (use_mixup or use_cutmix):
        return x, y, y, 1.0, None

    do_cutmix = use_cutmix and (not use_mixup or np.random.rand() < 0.5)
    lam = _sample_lam(cutmix_alpha if do_cutmix else mixup_alpha)
    idx = torch.randperm(B, device=x.device)
    y2 = y[idx]

    if do_cutmix:
        # Cutmix box
        r_x = np.random.randint(W)
        r_y = np.random.randint(H)
        r_w = int(W * np.sqrt(1 - lam))
        r_h = int(H * np.sqrt(1 - lam))
        x1 = np.clip(r_x - r_w // 2, 0, W)
        y1 = np.clip(r_y - r_h // 2, 0, H)
        x2 = np.clip(r_x + r_w // 2, 0, W)
        y2b = np.clip(r_y + r_h // 2, 0, H)
        x[:, :, y1:y2b, x1:x2] = x[idx, :, y1:y2b, x1:x2]
        lam = 1 - ((x2 - x1) * (y2b - y1) / (W * H))
        return x, y, y2, float(lam), "cutmix"
    else:
        x = lam * x + (1 - lam) * x[idx]
        return x, y, y2, float(lam), "mixup"

def _mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device="cuda", n_classes=20):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    top3_hits, total = 0, 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1)

        # confusion matrix
        y_np = y.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()
        for yt, yp in zip(y_np, p_np):
            cm[yt, yp] += 1

        # top-3
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
        f1.append(float(2 * prec * rec / (prec + rec + 1e-9)))

    macro_f1 = float(np.mean(f1))
    top3_acc = top3_hits / max(1, total)
    return {"acc": float(acc), "macro_f1": macro_f1, "per_class_f1": f1, "top3_acc": top3_acc, "cm": cm}

# ---------------------------
# Fit with checkpointing + resume
# ---------------------------
def fit_cls(
    index_csv_train: str,
    index_csv_val: str,
    in_ch: int = 61,
    n_classes: int = 20,
    bs: int = 32,
    lr: float = 3e-4,
    epochs: int = 30,
    num_workers: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    oversample: bool = True,
    monitor: str = "macro_f1",  # "macro_f1" or "acc"
    width: int = 64,
    seed: int = 42,
    # New stuff
    out_dir: Optional[str] = None,              # where to save logs/best
    ckpt_dir: Optional[str] = None,             # where to save checkpoints
    resume: bool = True,                         # auto-resume from latest
    resume_path: Optional[str] = None,          # or a specific ckpt
    label_smoothing: float = 0.05,
    mixup_alpha: float = 0.0,                   # e.g., 0.2 to enable mixup
    cutmix_alpha: float = 0.0,                  # e.g., 1.0 to enable cutmix
    grad_clip_norm: float = 1.0,
    ema_decay: float = 0.999,
    warmup_epochs: int = 3,
):
    """
    Returns: the *best* model (according to `monitor`), with EMA weights applied if EMA is used.
    """
    _seed_all(seed)
    device = _device()

    # I/O
    out_dir = _ensure_dir(out_dir or "./outputs")
    ckpt_dir = _ensure_dir(ckpt_dir or os.path.join(out_dir, "checkpoints"))
    logs_csv = os.path.join(out_dir, "train_log.csv")
    rng_state_path = os.path.join(ckpt_dir, "rng_state.json")
    best_weights_path = os.path.join(out_dir, "best.pt")

    # datasets
    train_ds = ParcelChipsCls(index_csv_train, normalize=True, augment=True)
    val_ds   = ParcelChipsCls(index_csv_val,   normalize=True, augment=False)

    # oversampling (based on train CSV)
    sampler = None
    if oversample:
        tr = pd.read_csv(index_csv_train)
        counts = tr["class_id"].value_counts().to_dict()
        # inverse-sqrt weighting is stable
        w_class = {int(k): 1.0 / math.sqrt(max(int(v), 1)) for k, v in counts.items()}
        weights = tr["class_id"].map(w_class).astype("float32").values
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    pin = (device == "cuda")
    train_ld = DataLoader(
        train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
    )
    val_ld = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
    )

    # model/optim
    model = ParcelCNN(in_ch=in_ch, n_classes=n_classes, width=width).to(device)

    # Loss (label smoothing + optional class weights)
    if class_weights is not None:
        assert class_weights.numel() == n_classes, \
            f"class_weights length must be {n_classes}, got {class_weights.numel()}"
        class_weights = class_weights.to(device)

    crit  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine with linear warmup
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / max(1, warmup_epochs)
        # cosine over remaining epochs
        progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # AMP + EMA
    is_cuda = (device == "cuda")
    amp_dtype = _amp_dtype()
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)
    ema = EMA(model, decay=ema_decay)

    start_epoch = 1
    best_metric = -1.0
    best_monitor_name = monitor

    # --------- Resume if requested ----------
    ckpt_to_load = None
    if resume_path is not None and os.path.isfile(resume_path):
        ckpt_to_load = resume_path
    elif resume:
        ckpt_to_load = _get_latest_checkpoint(ckpt_dir)

    if ckpt_to_load:
        print(f"[Resume] Loading checkpoint: {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and ckpt["ema"] is not None:
            # initialize EMA and load
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", best_metric)
        best_monitor_name = ckpt.get("monitor", best_monitor_name)
        # RNG states for exact resumption
        if os.path.isfile(rng_state_path):
            _load_rng_state(rng_state_path)
        print(f"[Resume] Starting at epoch {start_epoch} (best {best_monitor_name}={best_metric:.4f})")

    # --------- Metrics CSV header ----------
    if not os.path.isfile(logs_csv):
        pd.DataFrame([{
            "epoch": 0, "train_loss": np.nan, "val_acc": np.nan,
            "val_macro_f1": np.nan, "val_top3": np.nan, "lr": lr
        }]).to_csv(logs_csv, index=False)

    # training loop
    for e in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0

        bar = tqdm(train_ld, total=len(train_ld), desc=f"Epoch {e:03d}/{epochs}", leave=True)
        for i, (x, y) in enumerate(bar, 1):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # Mixup/Cutmix
            x_in, y_a, y_b, lam, mode = _apply_mixup_cutmix(x, y, mixup_alpha, cutmix_alpha)

            with torch.amp.autocast('cuda', enabled=is_cuda, dtype=amp_dtype):
                logits = model(x_in)
                if mode in ("mixup", "cutmix"):
                    loss = _mixup_criterion(crit, logits, y_a, y_b, lam)
                else:
                    loss = crit(logits, y)
            scaler.scale(loss).backward()

            # gradient clipping
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(opt); scaler.update()

            # EMA update
            ema.update(model)

            running += loss.item()
            bar.set_postfix(loss=f"{running/i:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        # Validation — evaluate with EMA weights for best signal
        # Swap weights to EMA for eval
        ema_backup = {k: p.detach().clone() for k, p in model.state_dict().items()}
        ema.copy_to(model)
        metrics = evaluate(model, val_ld, device=device, n_classes=n_classes)
        # restore original weights
        model.load_state_dict(ema_backup, strict=True)

        sched.step()

        msg = (f"Epoch {e:03d} | train loss {running/len(train_ld):.3f} | "
               f"val acc {metrics['acc']:.3f} | macroF1 {metrics['macro_f1']:.3f} | "
               f"top3 {metrics['top3_acc']:.3f}")
        print(msg)

        # Append to CSV
        pd.DataFrame([{
            "epoch": e,
            "train_loss": running/len(train_ld),
            "val_acc": metrics["acc"],
            "val_macro_f1": metrics["macro_f1"],
            "val_top3": metrics["top3_acc"],
            "lr": opt.param_groups[0]["lr"],
        }]).to_csv(logs_csv, mode="a", header=False, index=False)

        # Save per-epoch checkpoint (includes optimizer/scheduler/scaler/EMA)
        ckpt = {
            "epoch": e,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "best_metric": best_metric,
            "monitor": best_monitor_name,
            "config": {
                "in_ch": in_ch, "n_classes": n_classes, "width": width,
                "bs": bs, "lr": lr, "epochs": epochs, "mixup_alpha": mixup_alpha,
                "cutmix_alpha": cutmix_alpha, "label_smoothing": label_smoothing,
                "ema_decay": ema_decay, "oversample": oversample, "seed": seed
            }
        }
        _ensure_dir(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{e:03d}.pt")
        torch.save(ckpt, ckpt_path)
        # also update latest + RNG state
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        _save_rng_state(rng_state_path)

        # Track best (by monitor) — save EMA weights as best
        current = metrics["macro_f1"] if monitor == "macro_f1" else metrics["acc"]
        if current > best_metric:
            best_metric = current
            # store EMA weights as best
            ema.copy_to(model)
            torch.save(model.state_dict(), best_weights_path)
            # restore training weights
            model.load_state_dict(ema_backup, strict=True)

        # Optionally save confusion matrix per epoch (npy)
        np.save(os.path.join(out_dir, f"cm_epoch_{e:03d}.npy"), metrics["cm"])

    # Load best EMA weights before returning
    if os.path.isfile(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))

    return model
