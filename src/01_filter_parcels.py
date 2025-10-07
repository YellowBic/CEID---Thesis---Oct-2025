
import os, numpy as np, pandas as pd
import geopandas as gpd
import pyogrio, fiona

ROOT = "/content/drive/MyDrive/parcel_centric"
GDB  = os.path.join(ROOT, "dyt_ellada.gdb")
OUT  = os.path.join(ROOT, "data/parcels/parcels_filtered.gpkg")

# --- fixed class order (maps σε label_id 0..19) ---
CODE_ORDER = ["15","8","6","37","2","28.1","36.2","38","3.1","39",
              "45.2","21","24","66","18","67","19","41","45.1","7"]
ACCEPT = set(CODE_ORDER)
code_to_label = {c:i for i,c in enumerate(CODE_ORDER)}

# --- προτιμώμενο layer/field (override) ---
LAYER_OVERRIDE = "Parcels_2023"
FIELD_OVERRIDE = "Κωδικοί_Καλλιέργειας"

def to_code_str(v):
    if v is None: return ""
    if isinstance(v, (int, np.integer)):   return str(int(v))
    if isinstance(v, (float, np.floating)): return format(float(v), "g")  # π.χ. 28.1 όχι 28.100000
    return str(v).strip().replace(",", ".")

def list_layers(path):
    try:
        return list(fiona.listlayers(path))
    except Exception:
        ls = pyogrio.list_layers(path)
        if isinstance(ls, dict) and "name" in ls: return list(ls["name"])
        out=[]
        for it in (ls or []):
            if hasattr(it, "name"): out.append(it.name)
            elif isinstance(it, dict) and "name" in it: out.append(it["name"])
        return out

def detect_by_values(gdb, layer_names):
    best = (None, None, -1)
    for lname in layer_names:
        try:
            df = pyogrio.read_dataframe(gdb, layer=lname, max_features=20000)
            cols = [c for c in df.columns if c != "geometry"]
            for c in cols:
                vals = df[c].dropna().head(20000).map(to_code_str)
                hits = int(vals.isin(ACCEPT).sum())
                if hits > best[2]:
                    best = (lname, c, hits)
        except Exception as e:
            print(f"[warn] skip layer {lname}: {e}")
    if best[2] <= 0:
        raise RuntimeError("Δεν βρέθηκε στήλη με τους 20 κωδικούς καλλιέργειας.")
    print(f"Using by-value match: layer='{best[0]}' field='{best[1]}' | hits={best[2]}")
    return best[0], best[1]

# --- επιλογή layer/field ---
layers = list_layers(GDB)
if not layers:
    raise RuntimeError("Δεν βρέθηκαν layers στο GDB.")
print("Found layers:", layers)

use_layer, use_field = None, None
if LAYER_OVERRIDE in layers:
    schema = pyogrio.read_dataframe(GDB, layer=LAYER_OVERRIDE, max_features=0)
    if FIELD_OVERRIDE in schema.columns:
        use_layer, use_field = LAYER_OVERRIDE, FIELD_OVERRIDE
        print(f"Using overrides: layer='{use_layer}', field='{use_field}'")
    else:
        print(f"[info] FIELD_OVERRIDE '{FIELD_OVERRIDE}' δεν υπάρχει, fallback σε ανίχνευση.")
        use_layer, use_field = detect_by_values(GDB, layers)
else:
    print(f"[info] LAYER_OVERRIDE '{LAYER_OVERRIDE}' δεν υπάρχει, fallback σε ανίχνευση.")
    use_layer, use_field = detect_by_values(GDB, layers)

# --- διάβασμα & φιλτράρισμα ---
gdf = pyogrio.read_dataframe(GDB, layer=use_layer, columns=[use_field, "geometry"])
before = len(gdf)
gdf = gdf[gdf[use_field].notna()].copy()
gdf["crop_code"] = gdf[use_field].map(to_code_str)
gdf = gdf[gdf["crop_code"].isin(ACCEPT)].copy()

# --- mapping σε label_id & pid ---
gdf["label_id"] = gdf["crop_code"].map(code_to_label).astype("int16")
gdf = gdf.reset_index(drop=True)
gdf["pid"] = gdf.index.astype("int32")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
gdf.to_file(OUT, driver="GPKG")
print(f"Filtered parcels saved: {OUT} | rows: {len(gdf)} / {before}")
print("Per-class counts:", gdf["label_id"].value_counts().sort_index().to_dict())
