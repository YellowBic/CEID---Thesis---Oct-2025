# /content/drive/MyDrive/parcel_centric/src/05b_rebuild_splits_from_dir.py
import os, sys, numpy as np, pandas as pd

ROOT = os.environ.get("ROOT", "/content/drive/MyDrive/parcel_centric")
INDEX_CSV = os.path.join(ROOT, "indices/parcel_chips_cls.csv")
OUT_DIR   = os.path.join(ROOT, "indices")
os.makedirs(OUT_DIR, exist_ok=True)

# στόχοι ανά τάξη (default: 350/75/75)
TR_PER = int(os.environ.get("SPLIT_TRAIN_PER_CLASS", "350"))
VA_PER = int(os.environ.get("SPLIT_VAL_PER_CLASS",   "75"))
TE_PER = int(os.environ.get("SPLIT_TEST_PER_CLASS",  "75"))
SEED   = int(os.environ.get("SPLIT_SEED", "42"))

print(f"Targets per class: train={TR_PER}, val={VA_PER}, test={TE_PER} | seed={SEED}")

idx = pd.read_csv(INDEX_CSV)
assert {"npz_path","class_id","parcel_id"}.issubset(idx.columns), "❌ index CSV missing columns"

# κρατάμε ένα chip/parcel (αν τυχόν υπάρχουν διπλά)
idx = idx.drop_duplicates(subset=["parcel_id"]).reset_index(drop=True)

splits = []
rng = np.random.default_rng(SEED)
for cid, grp in idx.groupby("class_id"):
    grp = grp.sample(frac=1.0, random_state=SEED)  # shuffle
    need_tr, need_va, need_te = TR_PER, VA_PER, TE_PER
    n = len(grp)

    # κόβουμε με τη σειρά train/val/test μέχρι να τελειώσουν τα διαθέσιμα
    n_tr = min(need_tr, n)
    tr = grp.iloc[:n_tr]
    n_va = min(need_va, max(0, n - n_tr))
    va = grp.iloc[n_tr:n_tr+n_va]
    n_te = min(need_te, max(0, n - n_tr - n_va))
    te = grp.iloc[n_tr+n_va:n_tr+n_va+n_te]

    tr = tr.assign(split="train")
    va = va.assign(split="val")
    te = te.assign(split="test")

    splits.append(pd.concat([tr,va,te], ignore_index=True))

final = pd.concat(splits, ignore_index=True)
print("Split counts per class:")
print(final.groupby(["split","class_id"]).size().unstack(fill_value=0))

final[final["split"]=="train"][["npz_path","class_id","parcel_id"]].to_csv(os.path.join(OUT_DIR, "parcel_train.csv"), index=False)
final[final["split"]=="val"  ][["npz_path","class_id","parcel_id"]].to_csv(os.path.join(OUT_DIR, "parcel_val.csv"), index=False)
final[final["split"]=="test" ][["npz_path","class_id","parcel_id"]].to_csv(os.path.join(OUT_DIR, "parcel_test.csv"), index=False)

print("✅ Wrote splits →", OUT_DIR)
