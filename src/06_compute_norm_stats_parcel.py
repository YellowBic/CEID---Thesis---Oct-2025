
import os, numpy as np, pandas as pd
ROOT = "/content/drive/MyDrive/parcel_centric"
TR   = os.path.join(ROOT, "indices/parcel_train.csv")
OUT  = os.path.join(ROOT, "metadata/norm_stats_parcel.npz")

df = pd.read_csv(TR)
paths = df["npz_path"].tolist()

# για ταχύτητα: δείγμα chips και τυχαίων pixels
rng = np.random.default_rng(42)
max_files = min(1500, len(paths))
paths = list(rng.choice(paths, size=max_files, replace=False))

samples_per_chip = 2048
C = 60
vals = [ [] for _ in range(C) ]

for p in paths:
    d = np.load(p)
    x = d["x"].astype("float32")
    x = x[:60]  # μόνο τα 60 φασματικά
    H,W = x.shape[1:]
    idx = rng.integers(0, H*W, size=min(samples_per_chip, H*W))
    rr, cc = idx // W, idx % W
    for c in range(C):
        vals[c].extend( x[c, rr, cc].tolist() )

med = np.zeros(C, dtype=np.float32)
iqr = np.zeros(C, dtype=np.float32)
for c in range(C):
    v = np.array(vals[c], dtype=np.float32)
    if v.size == 0: continue
    q1, q2, q3 = np.percentile(v, [25,50,75])
    med[c] = q2; iqr[c] = max(1e-6, q3-q1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, median=med, iqr=iqr)
print("Saved:", OUT, "| channels:", C)
