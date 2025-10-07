
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class ParcelChipsCls(Dataset):
    def __init__(self, index_csv, normalize=True, stats_path="/content/drive/MyDrive/parcel_centric/metadata/norm_stats_parcel.npz", augment=True):
        self.df = pd.read_csv(index_csv)
        self.normalize = normalize
        self.augment = augment
        self.med = self.iqr = None
        if normalize:
            st = np.load(stats_path)
            self.med = st["median"].astype("float32")
            self.iqr = st["iqr"].astype("float32")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        d = np.load(self.df.iloc[i]["npz_path"])
        x = d["x"].astype("float32")             # (C,H,W) 60 or 61
        y = int(d["y_cls"][0])

        # normalize μόνο τα πρώτα 60 κανάλια
        if self.normalize and self.med is not None:
            x[:60] = (x[:60] - self.med[:,None,None]) / (self.iqr[:,None,None] + 1e-6)

        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
            k = np.random.randint(0,4)
            if k: x = np.rot90(x, k, axes=(1,2)).copy()

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
