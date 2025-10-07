# /content/drive/MyDrive/parcel_centric/src/dataset_parcel_rcnn.py
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class ParcelChipsRCNN(Dataset):
    """
    Returns:
      x: (T, C_in, H, W) with T=10, C_in=6 (or 7 if mask is appended)
      y: () class id
    """
    def __init__(self, index_csv,
                 normalize=True,
                 stats_path="/content/drive/MyDrive/parcel_centric/metadata/norm_stats_parcel.npz",
                 augment=True,
                 add_mask_channel=True):
        self.df = pd.read_csv(index_csv)
        self.normalize = normalize
        self.augment = augment
        self.add_mask = add_mask_channel

        self.med = self.iqr = None
        if self.normalize:
            st = np.load(stats_path)
            self.med = st["median"].astype("float32")  # (60,)
            self.iqr = st["iqr"].astype("float32")     # (60,)

    def __len__(self): return len(self.df)

    def _apply_aug(self, x):
        # x: (T, C, H, W) numpy float32
        if np.random.rand() < 0.5:
            x = x[:, :, :, ::-1].copy()   # horizontal
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1, :].copy()   # vertical
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(2,3)).copy()
        return x

    def __getitem__(self, i):
        d = np.load(self.df.iloc[i]["npz_path"])
        x_full = d["x"].astype("float32")     # (C, H, W), C=60 or 61
        y = int(d["y_cls"][0])

        H, W = x_full.shape[1:]
        # spectral first 60 channels -> (T=10, C=6, H, W)
        if x_full.shape[0] < 60:
            raise ValueError(f"Expected at least 60 spectral channels, got {x_full.shape[0]}")
        x60 = x_full[:60]
        T = x60.shape[0] // 6
        if T * 6 != 60:
            raise ValueError("Spectral channels not divisible into 10×6")

        x = x60.reshape(T, 6, H, W)

        # optional mask channel (replicated across time)
        if self.add_mask and x_full.shape[0] >= 61:
            mask = x_full[60:61]                        # (1, H, W)
            mask_t = np.repeat(mask[None, ...], T, axis=0)  # (T, 1, H, W)
            x = np.concatenate([x, mask_t], axis=1)     # (T, 7, H, W)

        # normalize ONLY the 60 spectral channels with median/IQR
        if self.normalize and self.med is not None:
            med = self.med.reshape(T, 6, 1, 1)          # (10,6,1,1)
            iqr = self.iqr.reshape(T, 6, 1, 1)
            x[:, :6] = (x[:, :6] - med) / (iqr + 1e-6)

        # same spatial augmentation for all timesteps
        if self.augment:
            x = self._apply_aug(x)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
