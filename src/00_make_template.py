
import os, glob, numpy as np, rasterio
from rasterio.enums import Resampling

ROOT = "/content/drive/MyDrive/parcel_centric"
TEMPLATE = os.path.join(ROOT, "data/grid/template.tif")

# βρες ένα B02 10m (ή B04) από την cache ή από το Drive
cands = glob.glob("/content/s2_cache/**/*_B02_10m.jp2", recursive=True) + \
        glob.glob(os.path.join(ROOT, "sentinel_data/**/*_B02_10m.jp2"), recursive=True)
if not cands:
    cands = glob.glob("/content/s2_cache/**/*_B04_10m.jp2", recursive=True) + \
            glob.glob(os.path.join(ROOT, "sentinel_data/**/*_B04_10m.jp2"), recursive=True)
assert cands, "Δεν βρέθηκε JP2 10m (B02/B04). Έλεγξε το sentinel_data."

ref_path = sorted(cands)[0]
with rasterio.open(ref_path) as src:
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255)
    blank = np.full((src.height, src.width), 255, dtype=np.uint8)

with rasterio.open(TEMPLATE, "w", **profile) as dst:
    dst.write(blank, 1)

print("Template:", TEMPLATE)
