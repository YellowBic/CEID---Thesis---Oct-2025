
import os, glob, numpy as np, rasterio
from rasterio.warp import reproject, Resampling

ROOT = "/content/drive/MyDrive/parcel_centric"
LOCAL_MONTHS = "/content/s2_cache"
TEMPLATE = os.path.join(ROOT, "data/grid/template.tif")
TMP_DIR  = "/content/s2_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

bands = ["B02_10m","B03_10m","B04_10m","B08_10m","B11_20m","B12_20m"]

def find_band(mdir, tag):
    hits = sorted(glob.glob(os.path.join(mdir, f"*_{tag}.jp2")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(mdir, f"*_{tag}.tif")))
    return hits[0] if hits else None

def month_dirs(root):
    return sorted([d for d in os.listdir(root) if d.startswith("month_") and os.path.isdir(os.path.join(root,d))])

with rasterio.open(TEMPLATE) as ref:
    H,W = ref.height, ref.width
    dst_tr, dst_crs = ref.transform, ref.crs

months = month_dirs(LOCAL_MONTHS)
assert months, f"Δεν βρέθηκαν month_* στο {LOCAL_MONTHS}"

for m in months:
    out = os.path.join(TMP_DIR, f"{m}.npy")
    if os.path.exists(out):
        print(f"[{m}] ok:", out); continue
    mdir = os.path.join(LOCAL_MONTHS, m)
    paths = {b: find_band(mdir,b) for b in bands}
    miss = [b for b,p in paths.items() if p is None]
    if miss: raise FileNotFoundError(f"{m} λείπουν: {miss}")

    arr = np.zeros((6,H,W), dtype=np.float32)

    # 10m bands copy/reproject if needed
    for i,b in enumerate(["B02_10m","B03_10m","B04_10m","B08_10m"]):
        with rasterio.open(paths[b]) as src:
            s = src.read(1)
            if s.shape != (H,W) or src.crs != dst_crs or src.transform != dst_tr:
                d = np.zeros((H,W), dtype=np.float32)
                reproject(s, d, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_tr, dst_crs=dst_crs, resampling=Resampling.bilinear)
                arr[i] = d
            else:
                arr[i] = s

    # 20m → 10m
    for b,idx in [("B11_20m",4),("B12_20m",5)]:
        with rasterio.open(paths[b]) as src:
            s = src.read(1)
            d = np.zeros((H,W), dtype=np.float32)
            reproject(s, d, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_tr, dst_crs=dst_crs, resampling=Resampling.bilinear)
            arr[idx] = d

    np.save(out, arr)
    print(f"[{m}] saved:", out, arr.shape)
print("Per-month arrays ready in", TMP_DIR)
