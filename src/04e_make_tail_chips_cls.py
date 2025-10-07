# /content/drive/MyDrive/parcel_centric/src/04e_make_tail_chips_cls.py
import os, re, sys, json, math, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import windows, features
from rasterio.transform import rowcol

ROOT = os.environ.get("ROOT", "/content/drive/MyDrive/parcel_centric")
PARCELS_GPKG = os.environ.get("PARCELS_GPKG", f"{ROOT}/data/parcels/parcels_filtered.gpkg")
TEMPLATE     = os.path.join(ROOT, "data/grid/template.tif")
TMP_DIR      = "/content/s2_tmp"  # month_XX.npy (6,H,W)
CHIPS_DIR    = os.path.join(ROOT, "data/parcel_chips_cls")
INDEX_CSV    = os.path.join(ROOT, "indices/parcel_chips_cls.csv")
os.makedirs(CHIPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)

# --- παραμέτροι μέσω env ---
TAIL_IDS   = os.environ.get("TAIL_IDS", "2,13,15,16,17")  # label_id (όχι crop_code)
TAIL_IDS   = [int(x) for x in TAIL_IDS.replace(" ","").split(",") if x!=""]
ADD_PER_CL = int(os.environ.get("TAIL_ADD_PER_CLASS", "200"))     # στόχος νέων chips / τάξη
MIN_COVER  = int(os.environ.get("TAIL_MIN_COVER_PIXELS", "40"))   # πιο χαλαρό από 200
CHIP       = int(os.environ.get("CHIP", "128"))
CONTEXT    = int(os.environ.get("CONTEXT", "8"))
ADD_MASK   = os.environ.get("ADD_MASK_CHANNEL", "1") == "1"

print(f"TAIL_IDS={TAIL_IDS} | ADD_PER_CL={ADD_PER_CL} | MIN_COVER={MIN_COVER} | CHIP={CHIP} | CONTEXT={CONTEXT} | MASK={ADD_MASK}")

# --- μήνες ---
month_re = re.compile(r"month_(\d{2})\.npy$")
months = sorted([f for f in os.listdir(TMP_DIR) if month_re.match(f)], key=lambda x: int(month_re.match(x).group(1)))
assert months, f"❌ Δεν βρέθηκαν month_XX.npy στο {TMP_DIR}"
m_arrs = [np.load(os.path.join(TMP_DIR,m), mmap_mode="r") for m in months]
T = len(m_arrs); B = 6
C = T*B + (1 if ADD_MASK else 0)

# --- template & parcels ---
with rasterio.open(TEMPLATE) as ref:
    H,W = ref.height, ref.width
    base_tr, crs = ref.transform, ref.crs

gdf = gpd.read_file(PARCELS_GPKG)
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

assert "label_id" in gdf.columns, "❌ λείπει 'label_id' στο parcels gpkg"
gdf = gdf[gdf["label_id"].isin(TAIL_IDS)].copy()
gdf = gdf.reset_index(drop=True)
if "pid" not in gdf.columns:
    gdf["pid"] = gdf.index.astype("int32")

# --- ήδη υπάρχοντα chips (για να αποφύγουμε διπλά) ---
existing = set()
if os.path.exists(INDEX_CSV):
    try:
        ex = pd.read_csv(INDEX_CSV)
        if "parcel_id" in ex.columns:
            existing = set(ex["parcel_id"].astype(int).tolist())
        else:
            # parse από filename: parcel_XXXXXXXX.npz
            existing = set([
                int(os.path.basename(p).split("_")[1].split(".")[0])
                for p in ex["npz_path"].astype(str).tolist()
                if os.path.basename(p).startswith("parcel_")
            ])
    except Exception:
        pass

print(f"Parcels διαθέσιμα για tails: {len(gdf)} | ήδη chips για {len(existing)} pids")

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

rows_new = []
added_per = {cid:0 for cid in TAIL_IDS}
saved = 0

for i, row in gdf.iterrows():
    cls = int(row["label_id"]); pid = int(row["pid"])
    if added_per[cls] >= ADD_PER_CL:
        continue
    if pid in existing:
        continue

    geom = row.geometry
    if geom is None or geom.is_empty:
        continue

    win = geom_window(geom)
    wtr = windows.transform(win, base_tr)
    mask = features.rasterize([(geom,1)], out_shape=(CHIP,CHIP), transform=wtr, fill=0, dtype="uint8")
    cover = int(mask.sum())
    if cover < MIN_COVER:
        continue

    r0,c0,h,w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
    x = np.empty((T*B, CHIP, CHIP), dtype=np.float32)
    for ti, arr in enumerate(m_arrs):
        x[ti*B:(ti+1)*B] = arr[:, r0:r0+h, c0:c0+w]
    # μηδενίζουμε εκτός parcel
    x *= (mask[None,:,:]>0).astype(np.float32)
    if ADD_MASK:
        x = np.concatenate([x, mask[None,:,:].astype(np.float32)], axis=0)

    npz_path = os.path.join(CHIPS_DIR, f"parcel_{pid:08d}.npz")
    if not os.path.exists(npz_path):
        np.savez_compressed(npz_path, x=x.astype("float16"), y_cls=np.array([cls],dtype=np.int16), pid=np.array([pid],dtype=np.int32))
        rows_new.append([npz_path, cls, pid, cover, x.shape[0]])
        existing.add(pid)
        added_per[cls] += 1
        saved += 1
        if saved % 200 == 0:
            print(f"Saved {saved} tail chips...")

# ενημέρωση index
if rows_new:
    cols = ["npz_path","class_id","parcel_id","cover_pixels","n_channels"]
    df_new = pd.DataFrame(rows_new, columns=cols)
    if os.path.exists(INDEX_CSV):
        df_old = pd.read_csv(INDEX_CSV)
        all_df = pd.concat([df_old, df_new], ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["parcel_id"], keep="first")
    else:
        all_df = df_new
    all_df.to_csv(INDEX_CSV, index=False)
    print(f"✅ Added {len(df_new)} new chips. Index updated → {INDEX_CSV} (total {len(all_df)})")
else:
    print("ℹ️ No new chips added (quota reached or no suitable parcels).")

print("Added per class:", added_per)
