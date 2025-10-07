
import os, re, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import windows, features
from rasterio.transform import rowcol
from shapely.geometry import box

ROOT = "/content/drive/MyDrive/parcel_centric"
TEMPLATE = os.path.join(ROOT, "data/grid/template.tif")
PARCELS  = os.path.join(ROOT, "data/parcels/parcels_filtered.gpkg")
TMP_DIR  = "/content/s2_tmp"
OUT_DIR  = os.path.join(ROOT, "data/parcel_chips_cls")
INDEX    = os.path.join(ROOT, "indices/parcel_chips_cls.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# σταθερές
CHIP = 128
CONTEXT = 8
MIN_COVER_PIXELS = 50  
ADD_MASK_CHANNEL = True
MAX_PER_CLASS = 500      # cap per class (None για άπειρα)
CODE_ORDER = ["15","8","6","37","2","28.1","36.2","38","3.1","39","45.2","21","24","66","18","67","19","41","45.1","7"]
code_to_label = {c:i for i,c in enumerate(CODE_ORDER)}

# months in order
month_re = re.compile(r"month_(\d{2})\.npy$")
months = sorted([f for f in os.listdir(TMP_DIR) if month_re.match(f)], key=lambda x: int(month_re.match(x).group(1)))
assert months, "Δεν βρέθηκαν month_XX.npy στο /content/s2_tmp"

with rasterio.open(TEMPLATE) as ref:
    H,W = ref.height, ref.width
    base_tr, crs = ref.transform, ref.crs

gdf = gpd.read_file(PARCELS)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)
assert "label_id" in gdf.columns, "λείπει 'label_id' στο parcels_filtered.gpkg"

# open months as memmaps
m_arrs = [np.load(os.path.join(TMP_DIR,m), mmap_mode="r") for m in months]
T = len(m_arrs); B = 6
C = T*B + (1 if ADD_MASK_CHANNEL else 0)

def geom_window(geom):
    minx, miny, maxx, maxy = geom.bounds
    r0,c0 = rowcol(base_tr, minx, maxy)
    r1,c1 = rowcol(base_tr, maxx, miny)
    r0 = max(0, r0 - CONTEXT); c0 = max(0, c0 - CONTEXT)
    r1 = min(H, r1 + CONTEXT); c1 = min(W, c1 + CONTEXT)
    rc, cc = (r0+r1)//2, (c0+c1)//2
    r_ul = max(0, min(H-CHIP, rc - CHIP//2))
    c_ul = max(0, min(W-CHIP, cc - CHIP//2))
    return windows.Window(c_ul, r_ul, CHIP, CHIP)

rows = []
per_class = {}
kept = 0

for i, row in gdf.iterrows():
    cls = int(row["label_id"])
    if MAX_PER_CLASS is not None and per_class.get(cls,0) >= MAX_PER_CLASS:
        continue
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue

    win = geom_window(geom)
    wtr = windows.transform(win, base_tr)

    # parcel mask in this window
    mask = features.rasterize([(geom,1)], out_shape=(CHIP,CHIP), transform=wtr, fill=0, dtype="uint8")
    cover = int(mask.sum())
    if cover < MIN_COVER_PIXELS:
        continue

    x = np.empty((T*B, CHIP, CHIP), dtype=np.float32)
    r0,c0,h,w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
    for ti, arr in enumerate(m_arrs):
        x[ti*B:(ti+1)*B] = arr[:, r0:r0+h, c0:c0+w]

    x *= (mask[None,:,:]>0).astype(np.float32)  # μηδενίζουμε ό,τι είναι εκτός parcel

    if ADD_MASK_CHANNEL:
        x = np.concatenate([x, mask[None,:,:].astype(np.float32)], axis=0)

    pid = int(row.get("pid", i))
    npz_path = os.path.join(OUT_DIR, f"parcel_{pid:08d}.npz")
    np.savez_compressed(npz_path, x=x.astype("float16"), y_cls=np.array([cls],dtype=np.int16), pid=np.array([pid],dtype=np.int32))

    rows.append([npz_path, cls, pid, cover, x.shape[0]])
    per_class[cls] = per_class.get(cls,0)+1
    kept += 1
    if kept % 500 == 0:
        print(f"Saved {kept} chips...")

import pandas as pd
pd.DataFrame(rows, columns=["npz_path","class_id","parcel_id","cover_pixels","n_channels"]).to_csv(INDEX, index=False)
print(f"Done. {kept} chips → {OUT_DIR}")
