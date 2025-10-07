
import os, pandas as pd, numpy as np
ROOT = "/content/drive/MyDrive/parcel_centric"
IDX  = os.path.join(ROOT, "indices/parcel_chips_cls.csv")

df = pd.read_csv(IDX)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

tr_rows, va_rows, te_rows = [], [], []
for cls, g in df.groupby("class_id"):
    n = len(g)
    n_tr = int(0.70*n)
    n_va = int(0.15*n)
    tr_rows.append(g.iloc[:n_tr])
    va_rows.append(g.iloc[n_tr:n_tr+n_va])
    te_rows.append(g.iloc[n_tr+n_va:])

train = pd.concat(tr_rows).sample(frac=1.0, random_state=42)
val   = pd.concat(va_rows).sample(frac=1.0, random_state=42)
test  = pd.concat(te_rows).sample(frac=1.0, random_state=42)

train.to_csv(os.path.join(ROOT,"indices/parcel_train.csv"), index=False)
val.to_csv(  os.path.join(ROOT,"indices/parcel_val.csv"),   index=False)
test.to_csv( os.path.join(ROOT,"indices/parcel_test.csv"),  index=False)
print("Splits:", len(train), len(val), len(test))
