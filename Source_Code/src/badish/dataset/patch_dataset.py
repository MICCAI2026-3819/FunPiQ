import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from badish.utils.registry import DatasetRegistry


@DatasetRegistry.register("PatchDataset")
class PatchDataset(Dataset):
    def __init__(self, patch_dataset_path: Path, balance: bool = False):
        z = np.load(patch_dataset_path)
        self.X = z["X"]
        self.y = z["y"]

        if balance:
            bad = self.X[self.y == 1]
            good = self.X[self.y == 0]
            n = min(len(bad), len(good))
            bad = bad[random.sample(range(len(bad)), n)]
            good = good[random.sample(range(len(good)), n)]
            self.X = np.vstack([bad, good])
            self.y = np.array([1] * n + [0] * n, dtype=np.uint8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@DatasetRegistry.register("patchdataset_pu")
class PatchDatasetPU(Dataset):
    def __init__(self, patch_dataset_path: Path, positive_class: int, split: str, dtype: str = "float32"):
        assert split in ["P", "U"], "split must be 'P' or 'U'"
        z = np.load(patch_dataset_path)
        X = z["X"]
        y = z["y"]

        pos_mask = y == positive_class
        idx = np.where(pos_mask)[0] if split == "P" else np.where(~pos_mask)[0]
        np_dtype = np.float32 if dtype == "float32" else np.float64
        self.X = np.asarray(X[idx], dtype=np_dtype, order="C")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])
