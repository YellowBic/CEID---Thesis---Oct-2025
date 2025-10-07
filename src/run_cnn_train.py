# /content/drive/MyDrive/parcel_centric/src/run_cnn_train.py
import os, sys, math, json, argparse, random, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

# project imports
THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path: sys.path.append(THIS)
from dataset_parcel_cls import ParcelChipsCls  # returns (x,y) or (x,y,meta)

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
            nn.MaxPool2d(2),   # -> 64x64

            nn.Conv2d(c, 2*c, 3, 1, 1, bias=False),   nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.Conv2d(2*c,2*c, 3, 1, 1, bias=False),  nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # -> 32x32

            nn.Conv2d(2*c,4*c, 3, 1, 1, bias=False),  nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4*c, n_classes)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))

# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def eval_metrics(model, loader, device="cuda", n_classes=20):
    model.eval()
    import numpy as np, torch
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    top3_hits, total = 0, 0
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        else:
            raise ValueError("Batch is not a tuple/list")

        x = x.to(device); y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)

        y_np = y.cpu().numpy(); p_np = pred.cpu().numpy()
        for yt, yp in zip(y_np, p_np):
            cm[yt, yp] += 1

        top3 = logits.topk(3, dim=1).indices
        top3_hits += (top3.eq(y[:, None])).any(1).sum().item()
        total += y.size(0)

    acc = cm.trace() / cm.sum().clip(min=1)
    f1 = []
    for k in range(n_classes):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1.append(float(2 * prec * rec / (prec + rec + 1e-9)))
    macro_f1 = float(np.mean(f1))
    top3_acc = top3_hits / max(1, total)
    return {"acc": float(acc), "macro_f1": macro_f1, "per_class_f1": f1, "top3": top3_acc, "cm": cm}

# ---------------------------
# Checkpoint utils
# ---------------------------
def save_ckpt(path, model, opt, sched, scaler, epoch, best_metric, monitor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt else None,
        "sched": sched.state_dict() if sched else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "monitor": monitor,
        "torch": torch.__version__,
    }, path)

def load_ckpt(path, model, opt=None, sched=None, scaler=None, map_location="cpu"):
    # PyTorch 2.6: default weights_only=True → μπλοκάρει unpickling
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # παλαιότερα torch δεν είχαν το weights_only kwarg
        ckpt = torch.load(path, map_location=map_location)
    except Exception:
        # allowlist για TorchVersion και retry (ασφαλές αν εμπιστεύεσαι το αρχείο σου)
        try:
            from torch.serialization import add_safe_globals
            import torch as _torch
            add_safe_globals([_torch.torch_version.TorchVersion])
            ckpt = torch.load(path, map_location=map_location)
        except Exception as e2:
            raise e2

    model.load_state_dict(ckpt["model"])
    if opt and ckpt.get("opt"): opt.load_state_dict(ckpt["opt"])
    if sched and ckpt.get("sched"): sched.load_state_dict(ckpt["sched"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    best_metric = float(ckpt.get("best_metric", -1.0))
    monitor = ckpt.get("monitor", "macro_f1")
    return epoch, best_metric, monitor


# ---------------------------
# Train one epoch
# ---------------------------
def train_one_epoch(model, loader, device, crit, opt, scaler, amp_dtype, grad_clip=None):
    model.train()
    running = 0.0
    is_cuda = (device == "cuda")
    bar = tqdm(loader, total=len(loader), leave=False, desc="train")
    for i, batch in enumerate(bar, 1):
        # batch: (x,y) or (x,y,meta)
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 2:
                x, y = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        else:
            raise ValueError("Batch is not a tuple/list")

        x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=is_cuda, dtype=amp_dtype):
            logits = model(x)
            loss = crit(logits, y)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update()
        running += loss.item()
        bar.set_postfix(loss=f"{running/i:.3f}")
    return running / max(1, len(loader))

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser("Parcel CNN trainer (epoch checkpoints + resume)")
    ap.add_argument("--root", type=str, required=True, help="Project root (parcel_centric)")
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv", type=str,   default=None)
    ap.add_argument("--in_ch", type=int, default=61)
    ap.add_argument("--n_classes", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--width", type=int, default=64, help="model base width (64/48/96...)")
    ap.add_argument("--monitor", type=str, default="macro_f1", choices=["macro_f1","acc"])
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    args = ap.parse_args()

    ROOT = args.root
    train_csv = args.train_csv or f"{ROOT}/indices/parcel_train.csv"
    val_csv   = args.val_csv   or f"{ROOT}/indices/parcel_val.csv"
    ckpt_dir  = args.ckpt_dir  or f"{ROOT}/outputs/checkpoints_cnn"
    out_dir   = args.out_dir   or f"{ROOT}/outputs/train_cnn"
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

    # seeds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | bs={args.bs} | workers={args.num_workers}")
    print("train_csv:", train_csv)
    print("val_csv  :", val_csv)
    sys.stdout.flush()

    # class weights (20-length) unless disabled
    class_weights = None
    if not args.no_class_weights:
        tr = pd.read_csv(train_csv)
        counts = tr["class_id"].value_counts().reindex(range(args.n_classes), fill_value=0).astype(int).values
        w = (1.0/np.sqrt(np.clip(counts,1,None))).astype("float32")
        class_weights = torch.tensor(w, device=device)
        print("Class counts:", dict(enumerate(counts)))

    # datasets
    train_ds = ParcelChipsCls(train_csv, normalize=True, augment=True)
    val_ds   = ParcelChipsCls(val_csv,   normalize=True, augment=False)

    # sampler (oversample)
    sampler = None
    if args.oversample:
        tr = pd.read_csv(train_csv)
        counts = tr["class_id"].value_counts().to_dict()
        w_class = {k: 1.0/math.sqrt(max(v,1)) for k,v in counts.items()}
        weights = tr["class_id"].map(w_class).astype("float32").values
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # dataloaders (with persistent workers & prefetch when workers>0)
    dl_common = dict(
        num_workers=args.num_workers,
        pin_memory=(device=="cuda"),
    )
    if args.num_workers > 0:
        dl_common.update(persistent_workers=True, prefetch_factor=4)

    train_ld = DataLoader(
        train_ds, batch_size=args.bs, shuffle=(sampler is None), sampler=sampler, **dl_common
    )
    val_ld = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False, **dl_common
    )

    # model/optim/amp
    model = ParcelCNN(in_ch=args.in_ch, n_classes=args.n_classes, width=args.width).to(device)
    crit  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    is_cuda = (device=="cuda")
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    # resume
    start_epoch, best_metric = 1, -1.0
    monitor = args.monitor
    last_ckpt = args.resume_path or os.path.join(ckpt_dir, "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        se, bm, mon = load_ckpt(last_ckpt, model, opt, sched, scaler, map_location=device)
        start_epoch = se + 1
        best_metric = bm
        monitor = mon or args.monitor
        print(f"Resumed from {last_ckpt} @ epoch {se} | best={best_metric:.4f} ({monitor})")

    # log header
    log_csv = os.path.join(out_dir, "log.csv")
    if (not args.resume) or (not os.path.exists(log_csv)):
        pd.DataFrame([["epoch","train_loss","val_acc","val_macro_f1","val_top3","lr"]]).to_csv(
            log_csv, index=False, header=False)

    # training loop
    for e in range(start_epoch, args.epochs+1):
        train_loss = train_one_epoch(model, train_ld, device, crit, opt, scaler, amp_dtype, grad_clip=args.grad_clip_norm)
        metrics = eval_metrics(model, val_ld, device=device, n_classes=args.n_classes)
        sched.step()

        lr_now = opt.param_groups[0]['lr']
        print(f"Epoch {e:03d}/{args.epochs} | train {train_loss:.3f} | "
              f"val acc {metrics['acc']:.3f} | macroF1 {metrics['macro_f1']:.3f} | top3 {metrics['top3']:.3f} | lr {lr_now:.2e}")

        with open(log_csv, "a") as f:
            f.write(f"{e},{train_loss:.6f},{metrics['acc']:.6f},{metrics['macro_f1']:.6f},{metrics['top3']:.6f},{lr_now:.6e}\n")

        # save last + best
        save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, opt, sched, scaler, e, best_metric, monitor)
        current = metrics["macro_f1"] if monitor=="macro_f1" else metrics["acc"]
        if current > best_metric:
            best_metric = current
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, None, None, None, e, best_metric, monitor)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_weights.pt"))

    print("Done. Checkpoints in:", ckpt_dir)
    print("Logs in:", log_csv)

if __name__ == "__main__":
    main()
